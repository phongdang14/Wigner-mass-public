from .ame2020      import parse             as parse_ame2020
from .ame2020      import parse_ame2016
from .nubase2020   import parse             as parse_nubase2020
from .data_manager import (fetch_ame2020, fetch_ame2016, fetch_nubase2020,
                            save_dataframe, load_dataframe, load_data,
                            clip, Normalizer)
from .merge        import build             as merge_tables
from .merge        import add_ame_source
from .su4xsu3      import add_su4_su3_columns
from .constants    import SHELLS_MAGIC, SHELLS_HO
from .plot         import plot_map, plot_maps, plot_predictions, plot_shap, plot_correlations
from .validate     import evaluate
from .training     import build_gk_indices, train, train_pinn
from .soap         import SOAP

__version__ = "0.1.0"

__all__ = [
    # I/O
    "parse_ame2020",
    "parse_ame2016",
    "parse_nubase2020",
    "fetch_ame2020",
    "fetch_ame2016",
    "fetch_nubase2020",
    "save_dataframe",
    "load_dataframe",
    "load_data",
    "clip",
    # Data utilities
    "Normalizer",
    "merge_tables",
    "add_ame_source",
    "add_su4_su3_columns",
    # Constants
    "SHELLS_MAGIC",
    "SHELLS_HO",
    # Plotting
    "plot_map",
    "plot_maps",
    "plot_predictions",
    "plot_shap",
    "plot_correlations",
    # Training
    "SOAP",
    "build_gk_indices",
    "train",
    "train_pinn",
    # Evaluation
    "evaluate",
]
