"""
SHAP-based feature attribution for the tabular branch.
"""

import numpy as np
import torch


def build_tabular_predictor(model, device: torch.device):
    """
    Return a numpy-in / numpy-out function wrapping the tabular branch
    of a CrossModalAttentionFusion model, suitable for shap.KernelExplainer.

    The imaging branch is zeroed out so SHAP attributions reflect
    only the tabular features' contribution to the risk score.

    Args:
        model  : CrossModalAttentionFusion instance (eval mode)
        device : torch device

    Returns:
        callable: (N, n_features) np.ndarray → (N,) np.ndarray
    """
    model.eval()

    def predict(tab_array: np.ndarray) -> np.ndarray:
        tab_tensor   = torch.tensor(tab_array, dtype=torch.float32).to(device)
        zero_img_feat = torch.zeros(tab_tensor.shape[0], 64, device=device)

        with torch.no_grad():
            f_tab = model.tab_encoder(tab_tensor)

            # direction 2: tabular attends to (zeroed) image
            Q2 = model.W_q2(f_tab).unsqueeze(1)
            K2 = model.W_k2(zero_img_feat).unsqueeze(1)
            V2 = model.W_v2(zero_img_feat).unsqueeze(1)
            a2 = torch.softmax(
                     torch.bmm(Q2, K2.transpose(1, 2)) / model.scale, dim=-1)
            f2 = torch.bmm(a2, V2).squeeze(1)

            # direction 1 approximation: use f_tab directly as attended output
            f1    = f_tab
            fused = torch.cat([f1, f2], dim=1)
            out   = model.head(fused).squeeze(-1)

        return out.cpu().numpy()

    return predict


def compute_shap_values(model,
                         test_tab:       np.ndarray,
                         background_tab: np.ndarray,
                         device:         torch.device,
                         nsamples:       int = 100):
    """
    Compute SHAP values for the tabular branch.

    Args:
        model          : CrossModalAttentionFusion (eval mode)
        test_tab       : (N_test, n_features) test tabular features
        background_tab : (N_bg, n_features)  background samples for SHAP
        device         : torch device
        nsamples       : number of SHAP samples per explanation

    Returns:
        shap_values : (N_test, n_features) SHAP value array
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    predictor   = build_tabular_predictor(model, device)
    explainer   = shap.KernelExplainer(predictor, background_tab)
    shap_values = explainer.shap_values(test_tab, nsamples=nsamples)
    return shap_values
