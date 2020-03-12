import numpy as np
import matplotlib.pyplot as plt


### base model ###
class Model:
    def deriv(self, state, t):
        raise NotImplementedError

    def get_var_names(self):
        raise NotImplementedError


### SIR ###
class SIR(Model):
    def S_dot_f(self, state, beta):
        return - beta * state[0] * state[1]

    def I_dot_f(self, state, alpha, beta):
        return beta * state[0] * state[1] - alpha * state[1]

    def R_dot_f(self, state, alpha):
        return alpha * state[1]

    def deriv(self, state, t, alpha, beta):
        S_dot = self.S_dot_f(state, beta)
        I_dot = self.I_dot_f(state, alpha, beta)
        R_dot = self.R_dot_f(state, alpha)
        N_dot = 0
        return np.asarray([S_dot, I_dot, R_dot, N_dot])

    def get_var_names(self):
        return ['susceptibles', 'infected', 'removed', 'total pop']


### SIR with deaths ###
class SIR_v2(Model):
    def S_dot_f(self, state, beta):
        return - beta * state[0] * state[1]

    def I_dot_f(self, state, alpha, beta):
        return beta * state[0] * state[1] - alpha * state[1]

    def R_dot_f(self, state, alpha, f):
        return f * alpha * state[1]

    def N_dot_f(self, state, alpha, f):
        return - alpha * (1 - f) * state[1]

    def deriv(self, state, t, alpha, beta, f):
        S_dot = self.S_dot_f(state, beta)
        I_dot = self.I_dot_f(state, alpha, beta)
        R_dot = self.R_dot_f(state, alpha, f)
        N_dot = self.N_dot_f(state, alpha, f)
        return np.asarray([S_dot, I_dot, R_dot, N_dot])

    def get_var_names(self):
        return ['susceptibles', 'infected', 'removed', 'total pop']


### SEIR ###
class SEIR(Model):
    def S_dot_f(self, state, beta, epsilon_E):
        infectious = epsilon_E * state[1] + state[2]
        outcoming = beta * state[0] * infectious
        return - outcoming 

    def E_dot_f(self, state, beta, epsilon_E, k):
        infectious = epsilon_E * state[1] + state[2]
        incoming = beta * state[0] * infectious
        outcoming = k * state[1]
        return incoming - outcoming

    def I_dot_f(self, state, alpha, beta, k):
        incoming = k * state[1]
        outcoming = alpha * state[2]
        return incoming - outcoming

    def R_dot_f(self, state, alpha, f):
        incoming = f * alpha * state[2]
        return incoming

    def N_dot_f(self, state, alpha, f):
        outcoming = alpha * (1 - f) * state[2]
        return - outcoming

    def deriv(self, state, t, alpha, beta, epsilon_E, k, f):
        S_dot = self.S_dot_f(state, beta, epsilon_E)
        E_dot = self.E_dot_f(state, beta, epsilon_E, k)
        I_dot = self.I_dot_f(state, alpha, beta, k)
        R_dot = self.R_dot_f(state, alpha, f)
        N_dot = self.N_dot_f(state, alpha, f)
        return np.asarray([S_dot, E_dot, I_dot, R_dot, N_dot])

    def get_var_names(self):
        return ['susceptibles', 'exposed', 'infected', 'removed', 'total pop']


### SITR ###
class SITR(Model):
    def S_dot_f(self, state, beta, delta):
        infectious = state[1] + delta * state[2]
        outcoming = beta * state[0] * infectious
        return - outcoming 

    def I_dot_f(self, state, alpha, beta, gamma, delta):
        infectious = state[1] + delta * state[2]
        incoming = beta * state[0] * infectious
        outcoming = (alpha + gamma) * state[1]
        return incoming - outcoming

    def T_dot_f(self, state, gamma, eta):
        incoming = gamma * state[1]
        outcoming = eta * state[2]
        return incoming - outcoming
        
    def R_dot_f(self, state, alpha, eta, f, fT):
        incoming = f * alpha * state[1] + fT * eta * state[2]
        return incoming

    def N_dot_f(self, state, alpha, eta, f, fT):
        outcoming = alpha * (1 - f) * state[1] + eta * (1 - fT) * state[2]
        return - outcoming

    def deriv(self, state, t, alpha, beta, gamma, delta, eta, f, fT):
        S_dot = self.S_dot_f(state, beta, delta)
        I_dot = self.I_dot_f(state, alpha, beta, gamma, delta)
        T_dot = self.T_dot_f(state, gamma, eta)
        R_dot = self.R_dot_f(state, alpha, eta, f, fT)
        N_dot = self.N_dot_f(state, alpha, eta, f, fT)
        return np.asarray([S_dot, I_dot, T_dot, R_dot, N_dot])

    def get_var_names(self):
        return ['susceptibles', 'infected', 'treated', 'removed', 'total pop']


### SEQIJR ###
class SEQIJR(Model):
    def S_dot_f(self, state, beta, epsilon_E, epsilon_Q, epsilon_J):
        infectious = epsilon_E * state[1] + epsilon_Q * state[2] \
                    + state[3] + epsilon_J * state[4]
        outcoming = beta * state[0] * infectious
        return - outcoming 

    def E_dot_f(self, state, beta, epsilon_E, epsilon_Q, epsilon_J, k_1, gamma_1):
        infectious = epsilon_E * state[1] + epsilon_Q * state[2] \
                    + state[3] + epsilon_J * state[4]
        incoming = beta * state[0] * infectious
        outcoming = (k_1 + gamma_1) * state[1]
        return incoming - outcoming

    def Q_dot_f(self, state, k_2, gamma_1):
        incoming = gamma_1 * state[1]
        outcoming = k_2 * state[2]
        return incoming - outcoming

    def I_dot_f(self, state, alpha_1, k_1, gamma_2):
        incoming = k_1 * state[1]
        outcoming = (alpha_1 + gamma_2) * state[3]
        return incoming - outcoming

    def J_dot_f(self, state, alpha_2, epsilon_J, k_2, gamma_2, f_2):
        incoming = k_2 * state[2] + gamma_2 * state[3]
        outcoming = alpha_2 * state[4]
        return incoming - outcoming

    def R_dot_f(self, state, alpha_1, alpha_2, f_1, f_2):
        incoming = alpha_1 * f_1 * state[3] + alpha_2 * f_2 * state[4]
        return incoming

    def N_dot_f(self, state, alpha_1, alpha_2, f_1, f_2):
        outcoming = alpha_1 * (1 - f_1) * state[3] + alpha_2 * (1 - f_2) * state[4]
        return - outcoming

    def deriv(self, state, t, alpha_1, alpha_2, beta, epsilon_E, epsilon_Q, epsilon_J,
              k_1, k_2, gamma_1, gamma_2, f_1, f_2):
        S_dot = self.S_dot_f(state, beta, epsilon_E, epsilon_Q, epsilon_J)
        E_dot = self.E_dot_f(state, beta, epsilon_E, epsilon_Q, epsilon_J, k_1, gamma_1)
        Q_dot = self.Q_dot_f(state, k_2, gamma_1)
        I_dot = self.I_dot_f(state, alpha_1, k_1, gamma_2)
        J_dot = self.J_dot_f(state, alpha_2, epsilon_J, k_2, gamma_2, f_2)
        R_dot = self.R_dot_f(state, alpha_1, alpha_2, f_1, f_2)
        N_dot = self.N_dot_f(state, alpha_1, alpha_2, f_1, f_2)
        return np.asarray([S_dot, E_dot, Q_dot, I_dot, J_dot, R_dot, N_dot])

    def get_var_names(self):
        return ['susceptibles', 'exposed', 'quarantined', 'infected', 
                'isolated', 'removed', 'total pop']