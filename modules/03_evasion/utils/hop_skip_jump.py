"""
HopSkipJump Attack Implementation

A decision-based adversarial attack that only requires access to the model's
predicted class (no confidence scores or gradients needed).

Reference: "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack"
           Chen et al., 2020
           https://arxiv.org/abs/1904.02144

This implementation is enhanced from the original Nvidia training with:
- Better documentation and type hints
- Progress tracking and query counting
- Configurable parameters
- Error handling
- Red team operational considerations
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable
from tqdm import tqdm


class HopSkipJump:
    """
    HopSkipJump attack for generating adversarial examples.
    
    This attack works in three main steps:
    1. Initialize: Find any misclassified sample
    2. Binary Search: Move to decision boundary
    3. Gradient Estimation: Estimate gradient via random sampling
    4. Boundary Walk: Iteratively improve the adversarial example
    
    Args:
        model: The target model to attack
        norm: Norm to minimize (2 for L2, np.inf for L-infinity)
        max_iter: Maximum number of iterations
        max_eval: Maximum number of model evaluations per iteration
        init_eval: Initial number of evaluations for gradient estimation
        init_size: Number of attempts to find initial adversarial
        clip_min: Minimum pixel value
        clip_max: Maximum pixel value
        verbose: Whether to print progress
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        norm: int = 2,
        max_iter: int = 64,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        clip_min: float = -2.0,
        clip_max: float = 2.0,
        verbose: bool = True
    ):
        self.model = model
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.verbose = verbose
        
        # Query tracking for operational security
        self.query_count = 0
        self.total_queries = 0
    
    def adversarial_satisfactory(
        self,
        samples: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if samples are adversarial (misclassified).
        
        Args:
            samples: Input samples to check
            target: Original class label
            
        Returns:
            Boolean tensor indicating which samples are adversarial
        """
        samples = torch.clamp(samples, self.clip_min, self.clip_max)
        
        with torch.no_grad():
            preds = self.model(samples).argmax(dim=1)
            self.query_count += len(samples)
        
        # Adversarial if prediction != target
        result = preds != target
        return result

    
    def compute_delta(
        self,
        current_sample: torch.Tensor,
        original_sample: torch.Tensor,
        theta: float,
        input_shape: Tuple,
        curr_iter: int
    ) -> float:
        """
        Compute the delta parameter for gradient estimation.
        
        Delta controls the size of perturbations used for gradient estimation.
        It adapts based on the current distance to the original sample.
        
        Args:
            current_sample: Current adversarial sample
            original_sample: Original input
            theta: Threshold parameter
            input_shape: Shape of input
            curr_iter: Current iteration number
            
        Returns:
            Delta value for this iteration
        """
        if curr_iter == 0:
            return 0.1 * (self.clip_max - self.clip_min)
        
        if self.norm == 2:
            dist = torch.norm(original_sample - current_sample)
            delta = torch.sqrt(torch.prod(torch.tensor(input_shape))) * theta * dist
        else:
            dist = torch.max(torch.abs(original_sample - current_sample))
            delta = torch.prod(torch.tensor(input_shape)) * theta * dist
        
        return delta.item()
    
    def binary_search(
        self,
        current_sample: torch.Tensor,
        original_sample: torch.Tensor,
        target: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Binary search to approach the decision boundary.
        
        Finds the point on the line between current_sample and original_sample
        that is closest to the decision boundary while remaining adversarial.
        
        Args:
            current_sample: Current adversarial sample
            original_sample: Original input
            target: Original class label
            threshold: Convergence threshold
            
        Returns:
            Sample on the decision boundary
        """
        upper_bound, lower_bound = 1.0, 0.0
        
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self.interpolate(
                current_sample=current_sample,
                original_sample=original_sample,
                alpha=alpha
            )
            
            # Check if still adversarial
            satisfied = self.adversarial_satisfactory(
                samples=interpolated_sample,
                target=target
            )
            
            # Update bounds
            if satisfied:
                upper_bound = alpha
            else:
                lower_bound = alpha
        
        result = self.interpolate(
            current_sample=current_sample,
            original_sample=original_sample,
            alpha=upper_bound
        )
        
        return result
    
    def compute_update(
        self,
        current_sample: torch.Tensor,
        num_eval: int,
        delta: float,
        target: torch.Tensor,
        input_shape: Tuple
    ) -> torch.Tensor:
        """
        Compute the gradient estimate via random sampling.
        
        This is the core of the attack: we estimate the gradient by:
        1. Sampling random perturbations
        2. Checking which ones remain adversarial
        3. Computing the mean direction that maintains adversarial status
        
        Args:
            current_sample: Current adversarial sample
            num_eval: Number of random samples to evaluate
            delta: Perturbation size
            target: Original class label
            input_shape: Shape of input
            
        Returns:
            Estimated gradient direction
        """
        # Generate random noise
        rnd_noise_shape = [num_eval] + list(input_shape)
        
        if self.norm == 2:
            rnd_noise = torch.randn(*rnd_noise_shape, device=current_sample.device)
        else:
            rnd_noise = torch.rand(*rnd_noise_shape, device=current_sample.device) * 2 - 1
        
        # Normalize random noise
        rnd_noise = rnd_noise / torch.sqrt(
            torch.sum(
                rnd_noise ** 2,
                dim=tuple(range(1, len(rnd_noise_shape))),
                keepdim=True
            )
        )
        
        # Create evaluation samples
        eval_samples = torch.clamp(
            current_sample + delta * rnd_noise,
            self.clip_min,
            self.clip_max
        )
        rnd_noise = (eval_samples - current_sample) / delta
        
        # Check which samples are still adversarial
        satisfied = self.adversarial_satisfactory(
            samples=eval_samples,
            target=target
        )
        
        # Convert to +1/-1
        f_val = 2 * satisfied.float().reshape([num_eval] + [1] * len(input_shape)) - 1.0
        
        # Compute gradient estimate
        if torch.mean(f_val) == 1.0:
            grad = torch.mean(rnd_noise, dim=0)
        elif torch.mean(f_val) == -1.0:
            grad = -torch.mean(rnd_noise, dim=0)
        else:
            f_val -= torch.mean(f_val)
            grad = torch.mean(f_val * rnd_noise, dim=0)
        
        # Normalize gradient
        if self.norm == 2:
            result = grad / torch.norm(grad)
        else:
            result = torch.sign(grad)
        
        return result
    
    def interpolate(
        self,
        current_sample: torch.Tensor,
        original_sample: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate between current and original samples.
        
        Args:
            current_sample: Current adversarial sample
            original_sample: Original input
            alpha: Interpolation parameter (0 = original, 1 = current)
            
        Returns:
            Interpolated sample
        """
        if self.norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = torch.clamp(
                current_sample,
                original_sample - alpha,
                original_sample + alpha
            )
        
        return result
    
    def attack(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Execute the HopSkipJump attack.
        
        Args:
            x: Input image(s) to attack
            y: True labels (if None, will be predicted)
            
        Returns:
            Tuple of (adversarial_examples, attack_info)
            attack_info contains: queries, distances, success
        """
        self.total_queries = 0
        
        # Get predictions if labels not provided
        if y is None:
            with torch.no_grad():
                y = self.model(x).argmax(dim=1)
        
        input_shape = x.squeeze(0).shape
        
        # Set binary search threshold
        if self.norm == 2:
            theta = 0.01 / np.sqrt(np.prod(input_shape))
        else:
            theta = 0.01 / np.prod(input_shape)
        
        results = []
        attack_info = {
            'queries': [],
            'distances': [],
            'success': []
        }
        
        # Attack each sample
        for ind in range(len(x)):
            self.query_count = 0
            original_sample = x[ind].unsqueeze(0)
            y_target = y[ind]
            
            if self.verbose:
                print(f"\n[*] Attacking sample {ind + 1}/{len(x)}")
            
            # Step 1: Find initial adversarial sample
            initial_sample = self._find_initial_adversarial(
                original_sample, y_target, theta
            )
            
            if initial_sample is None:
                if self.verbose:
                    print(f"[!] Failed to find initial adversarial sample")
                results.append(original_sample)
                attack_info['success'].append(False)
                attack_info['queries'].append(self.query_count)
                attack_info['distances'].append(float('inf'))
                continue
            
            # Step 2: Iteratively improve the adversarial example
            current_sample = initial_sample
            
            iterator = range(self.max_iter)
            if self.verbose:
                iterator = tqdm(iterator, desc="HSJ iterations")
            
            for curr_iter in iterator:
                # Compute delta for gradient estimation
                delta = self.compute_delta(
                    current_sample=current_sample,
                    original_sample=original_sample,
                    theta=theta,
                    input_shape=input_shape,
                    curr_iter=curr_iter
                )
                
                # Binary search to boundary
                current_sample = self.binary_search(
                    current_sample=current_sample,
                    original_sample=original_sample,
                    target=y_target,
                    threshold=theta
                )
                
                # Estimate gradient
                num_eval = min(
                    int(self.init_eval * np.sqrt(curr_iter + 1)),
                    self.max_eval
                )
                
                update = self.compute_update(
                    current_sample=current_sample,
                    num_eval=num_eval,
                    delta=delta,
                    target=y_target,
                    input_shape=input_shape
                )
                
                # Step size search
                if self.norm == 2:
                    dist = torch.norm(original_sample - current_sample)
                else:
                    dist = torch.max(torch.abs(original_sample - current_sample))
                
                epsilon = 2.0 * dist / np.sqrt(curr_iter + 1)
                success = False
                
                while not success:
                    epsilon /= 2.0
                    potential_sample = current_sample + epsilon * update
                    success = self.adversarial_satisfactory(
                        samples=potential_sample,
                        target=y_target
                    )
                
                # Update current sample
                current_sample = torch.clamp(
                    potential_sample,
                    self.clip_min,
                    self.clip_max
                )
                
                if self.verbose and curr_iter % 10 == 0:
                    iterator.set_postfix({
                        'dist': f'{dist.item():.4f}',
                        'queries': self.query_count
                    })
            
            # Final distance
            if self.norm == 2:
                final_dist = torch.norm(original_sample - current_sample).item()
            else:
                final_dist = torch.max(torch.abs(original_sample - current_sample)).item()
            
            results.append(current_sample)
            attack_info['success'].append(True)
            attack_info['queries'].append(self.query_count)
            attack_info['distances'].append(final_dist)
            
            if self.verbose:
                print(f"[+] Attack successful!")
                print(f"    Final distance: {final_dist:.4f}")
                print(f"    Total queries: {self.query_count}")
            
            self.total_queries += self.query_count
        
        adversarial_examples = torch.cat(results, dim=0)
        
        if self.verbose:
            print(f"\n[*] Attack complete")
            print(f"    Success rate: {sum(attack_info['success'])}/{len(x)}")
            print(f"    Total queries: {self.total_queries}")
            print(f"    Avg queries per sample: {self.total_queries / len(x):.0f}")
        
        return adversarial_examples, attack_info
    
    def _find_initial_adversarial(
        self,
        original_sample: torch.Tensor,
        y_target: torch.Tensor,
        theta: float
    ) -> Optional[torch.Tensor]:
        """
        Find an initial adversarial sample.
        
        Tries random samples until one is misclassified, then uses
        binary search to move it closer to the original.
        
        Args:
            original_sample: Original input
            y_target: Original class label
            theta: Threshold for binary search
            
        Returns:
            Initial adversarial sample, or None if not found
        """
        generator = torch.Generator(device=original_sample.device)
        generator.manual_seed(0)
        
        for attempt in range(self.init_size):
            # Generate random sample
            random_sample = torch.empty_like(original_sample).uniform_(
                self.clip_min,
                self.clip_max
            )
            
            # Check if misclassified
            if self.adversarial_satisfactory(random_sample, y_target):
                # Binary search to reduce distance
                initial_sample = self.binary_search(
                    current_sample=random_sample,
                    original_sample=original_sample,
                    target=y_target,
                    threshold=theta
                )
                
                if self.verbose:
                    print(f"[+] Found initial adversarial sample (attempt {attempt + 1})")
                
                return initial_sample
        
        return None


def hopskipjump_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """
    Convenience function for HopSkipJump attack.
    
    Args:
        model: Target model
        x: Input images
        y: True labels (optional)
        **kwargs: Additional arguments for HopSkipJump
        
    Returns:
        Tuple of (adversarial_examples, attack_info)
    
    Example:
        >>> adv_x, info = hopskipjump_attack(model, x, max_iter=50)
        >>> print(f"Success: {info['success']}")
        >>> print(f"Queries: {info['queries']}")
    """
    attacker = HopSkipJump(model, **kwargs)
    return attacker.attack(x, y)
