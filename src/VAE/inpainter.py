import sys
from pathlib import Path
import torch


_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from model import InpaintingVAE


class VAEInpainter:
    def __init__(self, ckpt_path, device="cuda", latent_dim=256, stochastic=True):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.stochastic = stochastic

        self.model = InpaintingVAE(latent_dim=latent_dim).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def inpaint(self, y, mask):
        return self.model.inpaint(y.to(self.device), mask.to(self.device),
                                   stochastic=self.stochastic)

    @torch.no_grad()
    def sample_diverse(self, y, mask, n_samples=4):
        return [self.model.inpaint(y.to(self.device), mask.to(self.device),
                                    stochastic=True) for _ in range(n_samples)]
