# ============================================================
# CELL 1: Environment Setup, Hardware Verification & Caching
# ============================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import copy
import gc
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings('ignore')

# ── Caching & Parallel Setup ──────────────────────────────
USE_CACHE = True
CACHE_DIR = './pipeline_cache'
RUNS_CACHE_DIR = os.path.join(CACHE_DIR, 'runs')
os.makedirs(RUNS_CACHE_DIR, exist_ok=True)

MAX_PARALLEL_JOBS = 10 
PRINT_LOCK = threading.Lock() # Prevents mangled console output

print(f"✅ Parallel   : {MAX_PARALLEL_JOBS} concurrent jobs")
print(f"✅ Caching    : {'Enabled' if USE_CACHE else 'Disabled'} (Dir: {RUNS_CACHE_DIR}/)")

# ── Hardware ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU        : {gpu_name}")
    print(f"✅ VRAM       : {vram_gb:.1f} GB")
else:
    print("⚠️  Running on CPU — GPU recommended")

print(f"✅ PyTorch    : {torch.__version__}")
print(f"✅ Device     : {device}")
# ============================================================


# ============================================================
# CELL 2: Data Pipeline (Shared Datasets)
# ============================================================
# We load datasets once globally, but instantiate DataLoaders
# locally per thread to ensure thread safety.
BATCH_SIZE  = 512

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std  = (0.2470, 0.2435, 0.2615)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,  download=True, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform)

CLASSES = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')

print(f"Train samples : {len(trainset):,}")
print(f"Test  samples : {len(testset):,}")
# ============================================================


# ============================================================
# CELL 3: PrunableLinear — Core Custom Layer
# ============================================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).cpu()

    @torch.no_grad()
    def get_effective_weights(self) -> torch.Tensor:
        return (self.weight * torch.sigmoid(self.gate_scores)).cpu()

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'params={self.in_features*self.out_features:,}')

print("✅ PrunableLinear defined")
# ============================================================


# ============================================================
# CELL 4: Network Architecture
# ============================================================
class SelfPruningNetwork(nn.Module):
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.flatten = nn.Flatten()

        self.bn1 = nn.BatchNorm1d(3072)
        self.fc1 = PrunableLinear(3072, 2048)
        self.drop1 = nn.Dropout(dropout_rate)

        self.bn2 = nn.BatchNorm1d(2048)
        self.fc2 = PrunableLinear(2048, 1024)
        self.drop2 = nn.Dropout(dropout_rate)

        self.bn3 = nn.BatchNorm1d(1024)
        self.fc3 = PrunableLinear(1024, 256)
        self.drop3 = nn.Dropout(dropout_rate)

        self.bn4 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.bn1(x)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.bn2(x)
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.bn3(x)
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.bn4(x)
        return self.fc4(x)

    def get_all_prunable_layers(self):
        return [(name, m) for name, m in self.named_modules()
                if isinstance(m, PrunableLinear)]

print("✅ SelfPruningNetwork defined")
# ============================================================


# ============================================================
# CELL 5: Loss Functions, Evaluation & Sparsity Metrics
# ============================================================
def calculate_sparsity_loss(model: nn.Module) -> torch.Tensor:
    l1 = torch.zeros(1, device=device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            l1 = l1 + torch.sigmoid(module.gate_scores).sum()
    return l1

def get_network_sparsity(model: nn.Module, threshold: float = 1e-2) -> dict:
    global_pruned, global_total = 0, 0
    layer_stats = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            clean_name = name.replace('_orig_mod.', '')
            if isinstance(module, PrunableLinear):
                gates   = torch.sigmoid(module.gate_scores)
                pruned  = (gates < threshold).sum().item()
                total   = gates.numel()
                sparsity = (pruned / total) * 100

                global_pruned += pruned
                global_total  += total
                layer_stats[clean_name] = {
                    'pruned'  : pruned, 'total'   : total,
                    'sparsity': sparsity, 'gates'   : gates.cpu().flatten().numpy()
                }

    global_sparsity = (global_pruned / global_total) * 100 if global_total else 0
    return {'global': global_sparsity, 'layers': layer_stats,
            'total_weights': global_total, 'total_pruned': global_pruned}

@torch.no_grad()
def evaluate(model: nn.Module, dataloader, criterion, lam: float) -> tuple:
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss    = criterion(outputs, labels) + lam * calculate_sparsity_loss(model)

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total, running_loss / len(dataloader)

def per_class_accuracy(model: nn.Module, dataloader) -> pd.DataFrame:
    model.eval()
    class_correct = {c: 0 for c in CLASSES}
    class_total   = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds.cpu()):
                c = CLASSES[label]
                class_total[c]   += 1
                class_correct[c] += (label == pred).item()

    rows = [(c, 100.0*class_correct[c]/class_total[c]) for c in CLASSES]
    return pd.DataFrame(rows, columns=['Class', 'Accuracy (%)'])

def calculate_fitness(acc: float, sparsity: float) -> float:
    if acc < 45.0: return 0.0  
    if acc < 50.0: return acc * (1.0 + 0.5 * sparsity / 100.0)
    return acc * (1.0 + sparsity / 100.0)

print("✅ Loss, eval, and fitness functions defined")
# ============================================================


# ============================================================
# CELL 6: Core Training Engine (Thread-Safe & Cached)
# ============================================================
def train_with_lambda(
    lam          : float,
    epochs       : int   = 80,
    patience     : int   = 10,
    lr           : float = 2e-3,
    weight_decay : float = 1e-4,
    dropout      : float = 0.3,
    verbose      : bool  = False,
    label        : str   = ""
) -> dict:
    
    # ── Check individual run cache first ──
    cache_path = os.path.join(RUNS_CACHE_DIR, f"run_lam_{lam:.6f}.pt")
    if USE_CACHE and os.path.exists(cache_path):
        with PRINT_LOCK:
            print(f"  💾 [CACHED] Restored {label} (λ={lam:.6f})")
        
        # Load from disk and reconstruct model object
        cached_result = torch.load(cache_path, map_location=device, weights_only=False)
        model = SelfPruningNetwork(dropout_rate=dropout).to(device)
        model.load_state_dict(cached_result['state_dict'])
        cached_result['model'] = model  # Re-attach live model for evaluation later
        return cached_result

    # ── Thread-Local DataLoaders ──
    # Important: 1 worker per thread prevents CPU bottlenecking across 10 jobs
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True, persistent_workers=True)
    testloader  = torch.utils.data.DataLoader(
        testset,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=True, persistent_workers=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = SelfPruningNetwork(dropout_rate=dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(trainloader),
        epochs=epochs, pct_start=0.1, anneal_strategy='cos'
    )

    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'sparsity': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs  = model(inputs)
                loss     = criterion(outputs, labels) + lam * calculate_sparsity_loss(model)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        val_acc, val_loss = evaluate(model, testloader, criterion, lam)
        sp_stats = get_network_sparsity(model)

        history['train_loss'].append(epoch_loss / n_batches)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['sparsity'].append(sp_stats['global'])

        if verbose and epoch % 5 == 0:
            with PRINT_LOCK:
                print(f"  {label} Ep {epoch+1:3d} | loss={val_loss:.4f} acc={val_acc:.1f}% sparsity={sp_stats['global']:.1f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose: 
                    with PRINT_LOCK: print(f"  {label} → Early stop epoch {epoch+1}")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    final_acc, _   = evaluate(model, testloader, criterion, lam)
    sp_stats_final = get_network_sparsity(model)
    fitness        = calculate_fitness(final_acc, sp_stats_final['global'])

    with PRINT_LOCK:
        print(f"  ✅ DONE {label} | λ={lam:.6f} | acc={final_acc:.2f}% | sp={sp_stats_final['global']:.1f}% | fit={fitness:.2f}")

    # Build result dict
    result = {
        'lam'       : lam,
        'acc'       : final_acc,
        'sparsity'  : sp_stats_final['global'],
        'fitness'   : fitness,
        'sp_stats'  : sp_stats_final,
        'history'   : history,
        'state_dict': best_state,
        'model'     : model
    }

    # ── Save individual run cache ──
    if USE_CACHE:
        cache_safe = result.copy()
        del cache_safe['model'] # Don't pickle live model objects, rely on state_dict
        torch.save(cache_safe, cache_path)

    return result

print("✅ Training engine defined (Parallel-safe & robustly cached)")
# ============================================================


# ============================================================
# CELL 7: Parallel Execution Wrapper
# ============================================================
def run_parallel_sweeps(lambdas, stage_name, epochs, patience, max_workers=MAX_PARALLEL_JOBS):
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_lam = {
            executor.submit(
                train_with_lambda, 
                lam=l, epochs=epochs, patience=patience, 
                verbose=False, label=f"[{stage_name} {i+1}/{len(lambdas)}]"
            ): l
            for i, l in enumerate(lambdas)
        }
        
        for future in as_completed(future_to_lam):
            lam = future_to_lam[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                with PRINT_LOCK:
                    print(f"❌ Exception for {stage_name} λ={lam:.6f}: {exc}")

    # Clean up GPU memory after large parallel execution
    gc.collect()
    torch.cuda.empty_cache()
    return results

print("✅ Parallel wrapper defined")
# ============================================================


# ============================================================
# CELL 8: Stage 1 — Coarse Exploration (Parallel)
# ============================================================
print("\n" + "=" * 65)
print("STAGE 1: Coarse Exploration (15 λ values in parallel)")
print("=" * 65)

coarse_lambdas = np.logspace(-5, np.log10(5e-2), 15)

# Launch all 15 concurrently (ThreadPool will cap at 10 simultaneously)
coarse_results = run_parallel_sweeps(
    coarse_lambdas, stage_name="Coarse", epochs=30, patience=7)

# Sort by fitness
coarse_df = pd.DataFrame([
    {'lam': r['lam'], 'acc': r['acc'], 'sparsity': r['sparsity'], 'fitness': r['fitness']}
    for r in coarse_results
]).sort_values('fitness', ascending=False)

top3_lams = coarse_df.head(3)['lam'].values
search_min = float(np.min(top3_lams)) / 3.0
search_max = float(np.max(top3_lams)) * 3.0

print(f"\n[CONVERGENCE] Top-3 λ values: {top3_lams}")
print(f"[CONVERGENCE] Fine search range: [{search_min:.6f}, {search_max:.6f}]")
print(f"\n{'λ':>12} {'Acc':>8} {'Sparsity':>10} {'Fitness':>10}")
print("-" * 44)
for _, row in coarse_df.iterrows():
    print(f"  {row['lam']:10.6f} {row['acc']:7.2f}% {row['sparsity']:9.2f}% {row['fitness']:9.2f}")


# ============================================================
# CELL 9: Stage 2 — Fine Exploitation (Parallel)
# ============================================================
print("\n" + "=" * 65)
print("STAGE 2: Fine Exploitation (50 λ values in parallel)")
print("=" * 65)

fine_lambdas = np.linspace(search_min, search_max, 50)

fine_results = run_parallel_sweeps(
    fine_lambdas, stage_name="Fine", epochs=80, patience=12)

# Find champion
best_result = None
best_fitness = 0.0
experiment_log = []

for i, r in enumerate(sorted(fine_results, key=lambda x: x['lam'])):
    experiment_log.append({
        'Experiment': i + 1, 'Lambda': r['lam'],
        'Accuracy': r['acc'], 'Sparsity': r['sparsity'], 'Fitness': r['fitness']
    })
    if r['fitness'] > best_fitness:
        best_fitness = r['fitness']
        best_result = r

exp_df = pd.DataFrame(experiment_log)

# Restore champion model
champion_model = best_result['model']
champion_model.eval()

print(f"\n{'='*65}")
print(f"SEARCH COMPLETE")
print(f"  Champion λ       : {best_result['lam']:.6f}")
print(f"  Test Accuracy    : {best_result['acc']:.2f}%")
print(f"  Global Sparsity  : {best_result['sparsity']:.2f}%")
print(f"  Fitness Score    : {best_fitness:.2f}")
print(f"{'='*65}")


# ============================================================
# CELL 10: Three Canonical Runs for Case Study
# ============================================================
print("\nTraining 3 canonical λ values for case study comparison table...")
print("(Low | Medium | High)\n")

canonical_lambdas = {
    'Low'   : 1e-5,
    'Medium': best_result['lam'],           
    'High'  : min(search_max * 2.0, 1e-1) 
}

# Run canonically. We do this sequentially to keep console output clean and verbose
canonical_results = {}
for tag, lam in canonical_lambdas.items():
    print(f"\nTraining Canonical Run: λ={lam:.6f} ({tag})...")
    res = train_with_lambda(
        lam=lam, epochs=100, patience=15, verbose=True, label=f"[{tag}]")
    canonical_results[tag] = res

table_rows = []
for tag, res in canonical_results.items():
    table_rows.append({
        'λ Label'    : tag,
        'Lambda (λ)' : f"{res['lam']:.6f}",
        'Test Accuracy (%)': f"{res['acc']:.2f}",
        'Global Sparsity (%)': f"{res['sparsity']:.2f}"
    })

comparison_df = pd.DataFrame(table_rows)
print("\n" + "="*60)
print("CASE STUDY TABLE: Lambda vs Accuracy vs Sparsity")
print("="*60)
print(comparison_df.to_string(index=False))


# ============================================================
# CELL 11: Production Hard-Prune & Verification
# ============================================================
@torch.no_grad()
def hard_prune_model(model: nn.Module, threshold: float = 1e-2) -> nn.Module:
    pruned_model = copy.deepcopy(model)
    raw = pruned_model._orig_mod if hasattr(pruned_model, '_orig_mod') else pruned_model
    replacements = {}

    for name, module in raw.named_modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            effective_w = module.weight * gates
            mask = (gates >= threshold).float()
            hard_w = effective_w * mask

            new_layer = nn.Linear(module.in_features, module.out_features, bias=True, device=device)
            new_layer.weight.data = hard_w
            new_layer.bias.data   = module.bias.data.clone()

            new_layer.weight.requires_grad_(False)
            new_layer.bias.requires_grad_(False)
            replacements[name] = new_layer

    for name, new_layer in replacements.items():
        parts = name.split('.')
        parent = raw
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)

    return pruned_model

@torch.no_grad()
def get_gate_distribution(model: nn.Module) -> np.ndarray:
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).cpu().flatten().numpy()
            all_gates.append(gates)
    return np.concatenate(all_gates) if all_gates else np.array([])


print(f"\nCHAMPION MODEL VERIFICATION (λ={best_result['lam']:.6f})")
print("=" * 65)

print("\n1. Layer-Wise Bottleneck Analysis:")
for name, stats in best_result['sp_stats']['layers'].items():
    print(f"   {name:<8} {stats['total']:>14,} params | {stats['sparsity']:>9.2f}% sparse")

print("\n2. Production Hard-Prune Deployment Test:")
production_model = hard_prune_model(champion_model, threshold=1e-2)

# Temporary dataloader just for evaluation
eval_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
prod_acc, _ = evaluate(production_model, eval_loader, nn.CrossEntropyLoss(), 0.0)
soft_acc    = best_result['acc']
deviation   = prod_acc - soft_acc

print(f"   Soft-gated model accuracy : {soft_acc:.2f}%")
print(f"   Hard-pruned model accuracy: {prod_acc:.2f}%")
print(f"   Accuracy deviation        : {deviation:+.2f}%")
if abs(deviation) < 2.0:
    print(f"   ✅ Minimal deviation — hard prune successful")

print("\n3. Per-Class Accuracy Breakdown:")
class_df = per_class_accuracy(champion_model, eval_loader)
print(class_df.to_string(index=False))

# ============================================================
# CELL 12: Visualisations
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

fig = plt.figure(figsize=(20, 28))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
all_gates = get_gate_distribution(champion_model)
near_zero = (all_gates < 0.01).sum() / len(all_gates) * 100
active    = (all_gates >= 0.1).sum()  / len(all_gates) * 100

ax1.hist(all_gates, bins=200, color='#2196F3', edgecolor='none', alpha=0.85)
ax1.set_yscale('log')
ax1.set_xlabel('Gate Value σ(g)', fontsize=13)
ax1.set_ylabel('Count (log scale)', fontsize=13)
ax1.set_title(
    f'Gate Value Distribution — Champion Model (λ={best_result["lam"]:.6f})\n'
    f'{near_zero:.1f}% gates near-zero (pruned) | {active:.1f}% active (≥0.1)',
    fontsize=14, fontweight='bold')
ax1.axvline(x=0.01,  color='red',    linestyle='--', lw=2, label='Prune threshold (0.01)')
ax1.axvline(x=0.5,   color='orange', linestyle='--', lw=1.5, label='Midpoint (0.5)')
ax1.legend(fontsize=11)

ax2 = fig.add_subplot(gs[1, 0])
for j, (name, stats) in enumerate(best_result['sp_stats']['layers'].items()):
    ax2.hist(stats['gates'], bins=100, alpha=0.6, color=COLORS[j % len(COLORS)],
             label=f'{name} (sp={stats["sparsity"]:.1f}%)', density=True)
ax2.set_yscale('log')
ax2.set_xlabel('Gate Value', fontsize=11)
ax2.set_ylabel('Density (log)', fontsize=11)
ax2.set_title('Layer-wise Gate Distributions', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[1, 1])
sc = ax3.scatter(
    exp_df['Sparsity'], exp_df['Accuracy'],
    c=exp_df['Fitness'], cmap='viridis', s=50, alpha=0.75, edgecolors='none')
ax3.scatter(
    best_result['sparsity'], best_result['acc'],
    color='red', marker='*', s=400, label='Champion', zorder=5)
plt.colorbar(sc, ax=ax3, label='Fitness Score')
ax3.set_xlabel('Sparsity Level (%)', fontsize=11)
ax3.set_ylabel('Test Accuracy (%)', fontsize=11)
ax3.set_title('Accuracy–Sparsity Pareto Curve', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)

ax4 = fig.add_subplot(gs[1, 2])
ax4.semilogx(coarse_df['lam'], coarse_df['fitness'],
             'o-', color='#2196F3', lw=2, ms=6, label='Coarse search')
ax4.axvline(x=search_min, color='green', linestyle='--', lw=1.5)
ax4.axvline(x=search_max, color='green', linestyle='--', lw=1.5)
ax4.axvline(x=best_result['lam'], color='red', linestyle='-', lw=2, label=f'Champion')
ax4.set_xlabel('λ (log scale)', fontsize=11)
ax4.set_ylabel('Fitness Score', fontsize=11)
ax4.set_title('Fitness vs λ', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)

for j, (tag, res) in enumerate(canonical_results.items()):
    ax = fig.add_subplot(gs[2, j])
    epochs_range = range(1, len(res['history']['val_acc']) + 1)
    ax.plot(epochs_range, res['history']['val_acc'], color=COLORS[j], lw=2, label='Val Accuracy')
    ax2_ = ax.twinx()
    ax2_.plot(epochs_range, res['history']['sparsity'], color=COLORS[j], lw=2, linestyle='--', alpha=0.7, label='Sparsity')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10, color=COLORS[j])
    ax2_.set_ylabel('Sparsity (%)', fontsize=10, color=COLORS[j])
    ax.set_title(f'λ={float(res["lam"]):.6f} ({tag})\nFinal: {res["acc"]:.1f}% acc | {res["sparsity"]:.1f}% sparse', fontsize=11, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='lower right')

ax8 = fig.add_subplot(gs[3, :])
tags  = list(canonical_results.keys())
accs  = [canonical_results[t]['acc'] for t in tags]
spars = [canonical_results[t]['sparsity'] for t in tags]
lams  = [f"λ={float(canonical_results[t]['lam']):.1e}\n({t})" for t in tags]

x = np.arange(len(tags))
w = 0.35
b1 = ax8.bar(x - w/2, accs,  w, color='#2196F3', alpha=0.85, label='Test Accuracy (%)')
b2 = ax8.bar(x + w/2, spars, w, color='#FF5722', alpha=0.85, label='Sparsity (%)')
ax8.bar_label(b1, fmt='%.1f%%', padding=3, fontsize=10)
ax8.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=10)
ax8.set_xticks(x)
ax8.set_xticklabels(lams, fontsize=11)
ax8.set_ylabel('Percentage (%)', fontsize=12)
ax8.set_title('λ Trade-off: Accuracy vs Sparsity', fontsize=13, fontweight='bold')
ax8.legend(fontsize=11)
ax8.set_ylim(0, 115)

fig.suptitle('The Self-Pruning Neural Network — Parallel Execution Results', fontsize=16, fontweight='bold', y=1.01)
plt.savefig('self_pruning_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Figure saved: self_pruning_results.png")


# ============================================================
# CELL 13: Inference Speed Benchmark & Saving
# ============================================================
def benchmark_throughput(model, n_iters=50, batch_size=512):
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(5): _ = model(x) # Warmup
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters): _ = model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - t0
    return (n_iters * batch_size) / elapsed

soft_tp = benchmark_throughput(champion_model)
hard_tp = benchmark_throughput(production_model)

print("\nINFERENCE THROUGHPUT BENCHMARK")
print("=" * 50)
print(f"  Soft-gated Throughput : {soft_tp:>10,.0f} samples/sec")
print(f"  Hard-pruned Throughput: {hard_tp:>10,.0f} samples/sec")
print(f"  Acceleration          : {hard_tp/soft_tp:.2f}x")

# Save outputs
torch.save({
    'model_state_dict': best_result['state_dict'],
    'lambda': best_result['lam'],
    'test_accuracy': best_result['acc'],
    'global_sparsity': best_result['sparsity'],
}, 'champion_model.pt')
torch.save(production_model.state_dict(), 'production_model.pt')
exp_df.to_csv('experiment_log.csv', index=False)

print("\n" + "="*65)
print("FINAL SUMMARY (PARALLEL EXECUTION)")
print("="*65)
print(f"  Champion λ          : {best_result['lam']:.6f}")
print(f"  Test Accuracy       : {best_result['acc']:.2f}%")
print(f"  Global Sparsity     : {best_result['sparsity']:.2f}%")
print(f"  Production Accuracy : {prod_acc:.2f}%")
print("="*65)