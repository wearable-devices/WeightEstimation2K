"""
Copyright (c) 2024-present NoteDance.
Apache-2.0 license
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class AdaBelief(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-16,
        weight_decay=0,
        amsgrad=False,
        decoupled_decay=True,
        fixed_decay=False,
        rectify=True,
        degenerated_to_sgd=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adabelief",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=None,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.weight_decay_ = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.decoupled_decay = decoupled_decay
        self.fixed_decay = fixed_decay
        self.rectify = rectify
        self.degenerated_to_sgd = degenerated_to_sgd
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.amsgrad = False
    
    def reset(self):
        for i,v in enumerate(self._trainable_variables):
            # State initialization
            self.step[i] = 0
            # Exponential moving average of gradient values
            self.exp_avg[i] = self.add_variable_from_reference(
                reference_variable=v, name="exp_avg"
            )

            # Exponential moving average of squared gradient values
            self.exp_avg_var[i] = self.add_variable_from_reference(
                reference_variable=v, name="exp_avg_var"
            )
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                self.max_exp_avg_var[i] = self.add_variable_from_reference(
                    reference_variable=v, name="max_exp_avg_var"
                )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_var = []
        if self.amsgrad:
            self.max_exp_avg_var = []
        self.buffer=[[None, None, None] for _ in range(10)]
        self.step = []
        for var in var_list:
            if var.dtype in {tf.float16, tf.bfloat16}:
                var_fp32 = tf.cast(var, 'float32')
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg"
                )
            )
            self.exp_avg_var.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg_var"
                )
            )
            if self.amsgrad:
                self.max_exp_avg_var.append(
                    self.add_variable_from_reference(
                        reference_variable=var_fp32, name="max_exp_avg_var"
                    )
                )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        if variable.dtype in {tf.float16, tf.bfloat16}:
            variable_fp32 = tf.cast(variable, 'float32')
        if gradient.dtype in {tf.float16, tf.bfloat16}:
            gradient = tf.cast(gradient, 'float32')
        
        # perform weight decay, check if decoupled weight decay
        if self.decoupled_decay:
            if not self.fixed_decay:
                variable_fp32.assign(variable_fp32 * (1.0 - lr * self.weight_decay_))
            else:
                variable_fp32.assign(variable_fp32 * (1.0 - self.weight_decay_))
        else:
            if self.weight_decay_ != 0:
                gradient.assign_add(self.weight_decay_ * variable_fp32)

        # get current state variable
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_var = self.exp_avg_var[self._get_variable_index(variable)]

        self.step[self._get_variable_index(variable)] += 1
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]

        # Update first and second moment running average
        exp_avg.assign(exp_avg * self.beta_1 + (1 - self.beta_1) * gradient)
        grad_residual = gradient - exp_avg
        exp_avg_var.assign(exp_avg_var * self.beta_2 + (1 - self.beta_2) * grad_residual * grad_residual)

        if self.amsgrad:
            max_exp_avg_var = self.max_exp_avg_var[self._get_variable_index(variable)]
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_var.assign(tf.maximum(max_exp_avg_var, exp_avg_var + self.epsilon))

            # Use the max. for normalizing running avg. of gradient
            denom = (tf.sqrt(max_exp_avg_var) / math.sqrt(bias_correction2)) + self.epsilon
        else:
            denom = (tf.sqrt(exp_avg_var + self.epsilon) / math.sqrt(bias_correction2)) + self.epsilon
        
        # update
        if not self.rectify:
            # Default update
            step_size = lr / bias_correction1
            variable_fp32.assign_add(-step_size * exp_avg / denom)
        else:
            # Rectified update, forked from RAdam
            buffered = self.buffer[int(self.step[self._get_variable_index(variable)] % 10)]
            if self.step[self._get_variable_index(variable)] == buffered[0]:
                num_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = self.step[self._get_variable_index(variable)]
                beta2_t = self.beta2 ** self.step[self._get_variable_index(variable)]
                num_sma_max = 2 / (1 - self.beta2) - 1
                num_sma = num_sma_max - 2 * self.step[self._get_variable_index(variable)] * beta2_t / (1 - beta2_t)
                buffered[1] = num_sma

                # more conservative since it's an approximated value
                if num_sma >= 5:
                    step_size = math.sqrt(
                        (1 - beta2_t) *
                        (num_sma - 4) / (num_sma_max - 4) *
                        (num_sma - 2) / num_sma *
                        num_sma_max / (num_sma_max - 2)) / (1 - self.beta1 ** self.step[self._get_variable_index(variable)])
                elif self.degenerated_to_sgd:
                    step_size = 1.0 / (1 - self.beta1 ** self.step[self._get_variable_index(variable)])
                else:
                    step_size = -1
                buffered[2] = step_size

            if num_sma >= 5:
                denom = tf.sqrt(exp_avg_var) + self.epsilon
                variable_fp32.assign_add(-step_size * lr * exp_avg / denom)
            elif step_size > 0:
                variable_fp32.assign_add(-step_size * lr * exp_avg)
        
        if variable.dtype in {tf.float16, tf.bfloat16}:
            variable.assign(variable_fp32)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay_,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "decoupled_decay": self.decoupled_decay,
                "fixed_decay": self.fixed_decay,
                "rectify": self.rectify,
                "degenerated_to_sgd": self.degenerated_to_sgd,
            }
        )
        return config