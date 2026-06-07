import json
import re

# Read the notebook
with open('demo/paper_kernels/linkedin/Multi_variate_Gaussian copy.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Standard format parameters to add
standard_params = """
    title_ads: List[str] = [],
    save: bool = False,
    name_ads: List[str] = [],
    font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
    save_types: List[str] = ['.png', '.svg'],
    paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'plot'},"""

# Find all cells with plotting functions
plot_cells = [6, 7, 8, 12, 13, 17, 18, 20, 29]

print(f"Found {len(plot_cells)} cells to update")

for cell_idx in plot_cells:
    cell = notebook['cells'][cell_idx]
    source = ''.join(cell.get('source', []))
    
    # Extract function name
    func_match = re.search(r'def\s+(\w+)', source)
    if func_match:
        func_name = func_match.group(1)
        print(f"\nProcessing cell {cell_idx}: {func_name}")
        
        # Check if it already has the standard format
        if 'title_ads: List[str]' in source or 'paper_figure:' in source:
            print(f"  Already has standard format, skipping")
            continue
        
        # This is a complex transformation - we'll need to manually update each function
        # For now, just report what needs to be updated
        print(f"  Needs updating")

print("\nDone analyzing. Manual updates needed for each function.")
