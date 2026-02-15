# Numerical Methods for a Nonlinear Autonomous ODE System

This repository contains a SageMath/Python implementation for the **analysis and numerical approximation of solutions** of a nonlinear autonomous system of three first-order ordinary differential equations:

$$
\begin{cases}
x' = \alpha (y - x - f(x)), \\
y' = x - y + z, \\
z' = -\beta y, \\
\alpha, \beta > 0
\end{cases}
$$


where the nonlinear function is defined as

\[
f(x) = m_1 x + \frac{m_0 - m_1}{2} \left( |x+1| - |x-1| \right).
\]

Due to the presence of absolute values in \( f(x) \), the system is **nonlinear and only piecewise differentiable**, which makes both the analytical and numerical study more challenging.

## Objectives

The main goals of this project are:

- Define and analyze the vector field associated with the system.
- Compute **equilibrium points**.
- Study the **linearization** around equilibrium points using the Jacobian matrix.
- Analyze **local stability** via eigenvalues.
- Approximate solutions of initial value problems (IVPs) using numerical methods.
- Compare different numerical schemes and study their errors.

## Methods Implemented

The following numerical and analytical methods are implemented:

- **Runge–Kutta (4th order)** method  
- **Taylor methods** (2nd and 4th order)  
- **Adams–Bashforth (3-step)** explicit multistep method  
- **Adams–Moulton / Predictor–Corrector** scheme  
- **Jacobian matrix computation** and eigenvalue analysis  
- **Piecewise analytical solution** on regions where the vector field is smooth  
- **Global and local truncation error estimation**
