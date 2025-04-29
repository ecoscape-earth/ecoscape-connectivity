    def forward(self, seed):
        """
        seed: a 0-1 (float) tensor of seed points.
        
        This is the "double coin toss" method. 
        """
        # First, we multiply the seed by the habitat, to confine the seeds to
        # where birds can live.
        # The multiplication by self.goodness is required for the gradient. 
        x = seed * self.habitat * self.goodness
        if x.ndim < 3:
            # We put it into shape (1, w, h) because the pooling operator expects this.
            x = torch.unsqueeze(x, dim=0)
        # Now we must propagate n times.
        zero = torch.zeros_like(x)
        mask = (x > 0) * 2
        for i in range(self.num_spreads):
            xx = x
            # Masks and randomizes the source.
            x = x * (mask > 0)
            mask = torch.maximum(mask - 1, zero)
            x = x * (self.min_transmission + (1. - self.min_transmission) * torch.rand_like(x))
            # Then, we propagate.
            x = self.spreader(x)
            # Second randomization. 
            x = x * (torch.rand_like(x) > 0.5)
            x = x * self.goodness
            # And finally we combine the results.
            mask = torch.maximum(2 * (x > xx + 0.05), mask)
            x = torch.maximum(x, xx)
            if torch.sum(mask) == 0:
                break
        x *= self.habitat
        if seed.ndim < 3:
            x = torch.squeeze(x, dim=0)
        return x

