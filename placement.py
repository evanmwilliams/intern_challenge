"""
VLSI Cell Placement Optimization - Hybrid Constructive + Gradient Approach
"""
import os
from enum import IntEnum
import torch
import torch.optim as optim

class CellFeatureIdx(IntEnum):
    AREA = 0
    NUM_PINS = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5

class PinFeatureIdx(IntEnum):
    CELL_IDX = 0
    PIN_X = 1
    PIN_Y = 2
    X = 3
    Y = 4
    WIDTH = 5
    HEIGHT = 6

MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0
STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0
MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_placement_input(num_macros, num_std_cells):
    total_cells = num_macros + num_std_cells
    macro_areas = torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))]
    areas = torch.cat([macro_areas, std_cell_areas])
    
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])
    
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)
    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()
    num_pins_per_cell[num_macros:] = torch.randint(MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,))
    
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights
    
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)
    PIN_SIZE = 0.1
    
    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.PIN_X] = pin_x
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.PIN_Y] = pin_y
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.X] = pin_x
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.Y] = pin_y
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx:pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE
        pin_idx += n_pins
    
    edge_list = []
    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx:pin_idx + n_pins] = cell_idx
        pin_idx += n_pins
    
    adjacency = [set() for _ in range(total_pins)]
    for pin_idx in range(total_pins):
        num_connections = torch.randint(1, 4, (1,)).item()
        for _ in range(num_connections):
            other_pin = torch.randint(0, total_pins, (1,)).item()
            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue
            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])
            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)
    
    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)
    
    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")
    return cell_features, pin_features, edge_list


def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)
    cell_positions = cell_features[:, 2:4]
    cell_indices = pin_features[:, 0].long()
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()
    dx = torch.abs(pin_absolute_x[src_pins] - pin_absolute_x[tgt_pins])
    dy = torch.abs(pin_absolute_y[src_pins] - pin_absolute_y[tgt_pins])
    alpha = 0.1
    smooth_manhattan = alpha * torch.logsumexp(torch.stack([dx / alpha, dy / alpha], dim=0), dim=0)
    return torch.sum(smooth_manhattan) / edge_list.shape[0]


def overlap_repulsion_loss(cell_features, pin_features, edge_list):
    """Vectorized overlap loss with softplus for smooth gradients."""
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)
    
    x = cell_features[:, 2]
    y = cell_features[:, 3]
    w = cell_features[:, 4]
    h = cell_features[:, 5]
    
    dx = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    dy = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))
    min_sep_x = (w.unsqueeze(0) + w.unsqueeze(1)) / 2
    min_sep_y = (h.unsqueeze(0) + h.unsqueeze(1)) / 2
    
    overlap_x = torch.nn.functional.softplus(min_sep_x - dx, beta=5.0)
    overlap_y = torch.nn.functional.softplus(min_sep_y - dy, beta=5.0)
    overlap_area = (overlap_x * overlap_y) ** 2
    
    mask = torch.triu(torch.ones(N, N, device=cell_features.device), diagonal=1)
    return (overlap_area * mask).sum() / N


def fast_legalize(cf, margin=1e-4, iters=100):
    """Iterative legalization - for small N only."""
    N = cf.shape[0]
    if N <= 1 or N > 500:  # Skip for large N - rely on shelf_pack
        return cf
    
    x = cf[:, CellFeatureIdx.X].clone().numpy()
    y = cf[:, CellFeatureIdx.Y].clone().numpy()
    w = cf[:, CellFeatureIdx.WIDTH].numpy()
    h = cf[:, CellFeatureIdx.HEIGHT].numpy()
    
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                min_sep_x = (w[i] + w[j]) / 2 + margin
                min_sep_y = (h[i] + h[j]) / 2 + margin
                pen_x = min_sep_x - dx
                pen_y = min_sep_y - dy
                
                if pen_x > 0 and pen_y > 0:
                    moved = True
                    if pen_x < pen_y:
                        push = pen_x / 2 + 1e-6
                        if x[i] < x[j]:
                            x[i] -= push
                            x[j] += push
                        else:
                            x[i] += push
                            x[j] -= push
                    else:
                        push = pen_y / 2 + 1e-6
                        if y[i] < y[j]:
                            y[i] -= push
                            y[j] += push
                        else:
                            y[i] += push
                            y[j] -= push
        if not moved:
            break
    
    out = cf.clone()
    out[:, CellFeatureIdx.X] = torch.tensor(x)
    out[:, CellFeatureIdx.Y] = torch.tensor(y)
    return out


def shelf_pack(cf, margin=1e-4):
    """Pack cells into rows sorted by height descending. Guarantees zero overlap."""
    N = cf.shape[0]
    w = cf[:, CellFeatureIdx.WIDTH]
    h = cf[:, CellFeatureIdx.HEIGHT]
    area = cf[:, CellFeatureIdx.AREA]
    
    # Sort by height descending (macros first), then area
    order = torch.argsort(h * 1e6 + area, descending=True)
    total_area = (w * h).sum().item()
    target_width = (total_area ** 0.5) * 1.2
    
    new_x = torch.zeros(N)
    new_y = torch.zeros(N)
    cur_x, cur_y, shelf_h = 0.0, 0.0, 0.0
    
    for idx in order:
        wi = w[idx].item()
        hi = h[idx].item()
        
        # Start new row if this cell doesn't fit
        if cur_x > 0 and cur_x + wi + margin > target_width:
            cur_x = 0.0
            cur_y += shelf_h + margin
            shelf_h = 0.0
        
        # Place cell (position is center)
        new_x[idx] = cur_x + wi / 2
        new_y[idx] = cur_y + hi / 2
        cur_x += wi + margin
        shelf_h = max(shelf_h, hi)
    
    out = cf.clone()
    out[:, CellFeatureIdx.X] = new_x
    out[:, CellFeatureIdx.Y] = new_y
    return out


def build_cell_adjacency(pin_features, edge_list):
    if edge_list.shape[0] == 0:
        return []
    cell_idx = pin_features[:, PinFeatureIdx.CELL_IDX].long()
    num_cells = int(cell_idx.max().item()) + 1
    nbrs = [set() for _ in range(num_cells)]
    src = cell_idx[edge_list[:, 0].long()]
    dst = cell_idx[edge_list[:, 1].long()]
    for a, b in zip(src.tolist(), dst.tolist()):
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    return nbrs


def barycentric_refine(cf, pin_features, edge_list, passes=3, margin=1e-4):
    """Move cells toward neighbor centroids to reduce wirelength."""
    N = cf.shape[0]
    nbrs = build_cell_adjacency(pin_features, edge_list)
    if not nbrs:
        return cf
    
    for _ in range(passes):
        x = cf[:, CellFeatureIdx.X].clone()
        y = cf[:, CellFeatureIdx.Y].clone()
        new_x, new_y = x.clone(), y.clone()
        
        for i, ns in enumerate(nbrs):
            if ns:
                new_x[i] = sum(x[j].item() for j in ns) / len(ns)
                new_y[i] = sum(y[j].item() for j in ns) / len(ns)
        
        cf = cf.clone()
        cf[:, CellFeatureIdx.X] = new_x
        cf[:, CellFeatureIdx.Y] = new_y
        
        # Re-legalize: for large N use shelf_pack, for small N use iterative
        if N > 500:
            cf = shelf_pack(cf, margin=margin)
        else:
            cf = fast_legalize(cf, margin=margin, iters=30)
    return cf


def train_placement(
    cell_features, pin_features, edge_list,
    num_epochs=1000, lr=0.1, lambda_wirelength=1.0, lambda_overlap=100.0,
    verbose=True, log_interval=100,
):
    """Hybrid: constructive placement + optional gradient refinement."""
    N = cell_features.shape[0]
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()
    
    # Phase 1: Constructive placement (guarantees zero overlap)
    cell_features = shelf_pack(cell_features, margin=1e-4)
    
    # Phase 2: Barycentric refinement for wirelength (fewer passes for large N)
    passes = 2 if N > 1000 else 3
    cell_features = barycentric_refine(cell_features, pin_features, edge_list, passes=passes, margin=1e-4)
    
    # Phase 3: Gradient refinement for small designs only
    if N <= 300:
        cell_positions = cell_features[:, 2:4].clone().detach().requires_grad_(True)
        optimizer = optim.Adam([cell_positions], lr=lr)
        epochs = min(num_epochs, 500)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            cf_current = cell_features.clone()
            cf_current[:, 2:4] = cell_positions
            
            wl_loss = wirelength_attraction_loss(cf_current, pin_features, edge_list)
            ol_loss = overlap_repulsion_loss(cf_current, pin_features, edge_list)
            
            progress = epoch / epochs
            total_loss = lambda_wirelength * progress * wl_loss + lambda_overlap * (1 + 10 * (1 - progress)) * ol_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)
            optimizer.step()
            
            if epoch % 50 == 0:
                with torch.no_grad():
                    tmp = cell_features.clone()
                    tmp[:, 2:4] = cell_positions
                    tmp = fast_legalize(tmp, margin=1e-4, iters=20)
                    cell_positions.copy_(tmp[:, 2:4])
            
            if verbose and epoch % log_interval == 0:
                print(f"Epoch {epoch}: OL={ol_loss.item():.6f} WL={wl_loss.item():.4f}")
        
        cell_features[:, 2:4] = cell_positions.detach()
        cell_features = fast_legalize(cell_features, margin=1e-4, iters=100)
    
    return {
        "final_cell_features": cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": {"total_loss": [], "wirelength_loss": [], "overlap_loss": []},
    }


# ======= EVALUATION CODE =======

def calculate_overlap_metrics(cell_features):
    N = cell_features.shape[0]
    if N <= 1:
        return {"overlap_count": 0, "total_overlap_area": 0.0, "max_overlap_area": 0.0, "overlap_percentage": 0.0}
    
    positions = cell_features[:, 2:4].detach().numpy()
    widths = cell_features[:, 4].detach().numpy()
    heights = cell_features[:, 5].detach().numpy()
    areas = cell_features[:, 0].detach().numpy()
    
    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
    
    total_area = sum(areas)
    overlap_percentage = (overlap_count / N * 100) if total_area > 0 else 0.0
    return {"overlap_count": overlap_count, "total_overlap_area": total_overlap_area,
            "max_overlap_area": max_overlap_area, "overlap_percentage": overlap_percentage}


def calculate_cells_with_overlaps(cell_features):
    N = cell_features.shape[0]
    if N <= 1:
        return set()
    
    positions = cell_features[:, 2:4].detach().numpy()
    widths = cell_features[:, 4].detach().numpy()
    heights = cell_features[:, 5].detach().numpy()
    cells_with_overlaps = set()
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2
            if max(0, min_sep_x - dx) > 0 and max(0, min_sep_y - dy) > 0:
                cells_with_overlaps.add(i)
                cells_with_overlaps.add(j)
    return cells_with_overlaps


def calculate_normalized_metrics(cell_features, pin_features, edge_list):
    N = cell_features.shape[0]
    cells_with_overlaps = calculate_cells_with_overlaps(cell_features)
    num_cells_with_overlaps = len(cells_with_overlaps)
    overlap_ratio = num_cells_with_overlaps / N if N > 0 else 0.0
    
    if edge_list.shape[0] == 0:
        return {"overlap_ratio": overlap_ratio, "normalized_wl": 0.0,
                "num_cells_with_overlaps": num_cells_with_overlaps, "total_cells": N, "num_nets": 0}
    
    wl_loss = wirelength_attraction_loss(cell_features, pin_features, edge_list)
    total_wirelength = wl_loss.item() * edge_list.shape[0]
    total_area = cell_features[:, 0].sum().item()
    num_nets = edge_list.shape[0]
    normalized_wl = (total_wirelength / num_nets) / (total_area ** 0.5) if total_area > 0 else 0.0
    
    return {"overlap_ratio": overlap_ratio, "normalized_wl": normalized_wl,
            "num_cells_with_overlaps": num_cells_with_overlaps, "total_cells": N, "num_nets": num_nets}


def plot_placement(initial_cell_features, final_cell_features, pin_features, edge_list, filename="placement_result.png"):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        for ax, cell_features, title in [(ax1, initial_cell_features, "Initial"), (ax2, final_cell_features, "Final")]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().numpy()
            widths = cell_features[:, 4].detach().numpy()
            heights = cell_features[:, 5].detach().numpy()
            for i in range(N):
                rect = Rectangle((positions[i, 0] - widths[i]/2, positions[i, 1] - heights[i]/2),
                                  widths[i], heights[i], fill=True, facecolor="lightblue",
                                  edgecolor="darkblue", linewidth=0.5, alpha=0.7)
                ax.add_patch(rect)
            metrics = calculate_overlap_metrics(cell_features)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{title}\nOverlaps: {metrics['overlap_count']}")
            ax.set_xlim(positions[:, 0].min() - 10, positions[:, 0].max() + 10)
            ax.set_ylim(positions[:, 1].min() - 10, positions[:, 1].max() + 10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def main():
    torch.manual_seed(42)
    cell_features, pin_features, edge_list = generate_placement_input(3, 50)
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    
    result = train_placement(cell_features, pin_features, edge_list, verbose=True, log_interval=200)
    metrics = calculate_normalized_metrics(result["final_cell_features"], pin_features, edge_list)
    print(f"\nFinal: Overlap={metrics['overlap_ratio']:.4f} WL={metrics['normalized_wl']:.4f}")
    plot_placement(result["initial_cell_features"], result["final_cell_features"], pin_features, edge_list)

if __name__ == "__main__":
    main()
