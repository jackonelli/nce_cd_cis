import torch

from src.models.base_model import BaseModel
from src.models.ace.ace_model import ResidualBlock


class AceProposal(BaseModel):
    def __init__(self, num_features: int, num_context_units: int = 64, num_components: int = 10,
                 num_blocks: int = 4, num_hidden_units: int = 512, activation: str = "relu", dropout_rate: float = 0.0,
                 **kwargs): # TODO: ta bort kwargs?

        super(AceProposal, self).__init__()

        self.num_features = num_features
        self.num_components = num_components
        self.num_context_units = num_context_units

        # TODO: handle this in nicer way (or always use relu)
        if activation == "relu":
            self.activation_fun = torch.nn.ReLU
        else:
            print("unknown activation")
            self.activation_fun = torch.nn.ReLU

        self.input_layer = torch.nn.Linear(in_features=2*self.num_features, out_features=num_hidden_units)
        # h = tfl.Dense(hidden_units)(h)

        self.num_blocks = num_blocks
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(num_hidden_units, num_hidden_units, self.activation_fun,
                                                                  dropout_rate) for _ in range(num_blocks)])

        self.output_layer = torch.nn.Linear(num_hidden_units, num_features * (3 * num_components + num_context_units))

    def sample(self, size: torch.Size, x):
        # Note: expect x to be a tuple (observed input, observed_mask)

        proposal_distr, _ = self.forward(x)

        return self.inner_sample(proposal_distr, size) # TODO: Need to check so that size is correct

    def inner_sample(self, distr, size):
        return distr.sample(size).transpose(0, 1)

    def forward(self, x: tuple):
        # Note: expect x to be a tuple (observed input, observed_mask)
        x_o, mask = x

        h = torch.cat((x_o, mask), dim=-1)
        x = self.activation_fun()(self.input_layer(h))

        for module in self.residual_blocks:
            x = module(x)

        x = self.output_layer(x).reshape(-1, self.num_features, 3 * self.num_components + self.num_context_units)

        context = x[..., :self.num_context_units]
        params = x[..., self.num_context_units:]

        proposal_distr = self.create_proposal_distr(params)

        return proposal_distr, context

    def log_prob(self, samples, x=0):
        # Note: expect x to be a tuple (observed input, observed_mask)

        # Calculate log prob conditionen on x
        proposal_distr, _ = self.forward(samples)

        return self.inner_log_prob(proposal_distr, samples).reshape()

    def inner_log_prob(self, distr, samples):
        return distr.log_prob(samples)

    def create_proposal_distr(self, params, eps=1e-3):
        logits = params[..., :self.num_components]
        means = params[..., self.num_components:-self.num_components]
        scales = torch.nn.Softplus()(params[..., -self.num_components:]) + eps

        components_distr = torch.distributions.Normal(loc=means, scale=scales)

        # TODO: does this work with gradients?
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(logits=logits),
            component_distribution=components_distr)

    def prob(self, samples):
        """Probability of a sample y"""
        return torch.exp(self.inner_log_prob(samples))
