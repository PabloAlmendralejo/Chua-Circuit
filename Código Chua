# Partiendo de la sistema de 3 ecuaciones diferenciales ordinarias de primer orden autónomas $$ $$ $$  x'=\alpha(y-x-f(x)),\quad y'= x-y+z, \quad z'=-\beta y \quad \alpha,\beta>0 \quad \\ $$
# donde $$ f(x)=m_1x + \frac{m_0 - m_1}{2}(|x+1|-|x-1|) $$ 
# $$\\ $$Y cuyo campo viene determinado por la función: $$$$ $F: \mathbb{R}^4 \to \mathbb{R}^3 \quad (x(t), y(t), z(t), t) \mapsto (\alpha(y(t) - x(t) - f(x(t)), x(t) - y(t) + z(t), -\beta y(t)) $ $ \\ $
# Se busca aproximar las soluciones de los p.v.i de dicho sistema a traves de métodos numéricos
# 
# El hecho de que la función $f(x)$ contenga un valor absoluto hace que el sistema de ecuaciones difereniales no sea lineal. $ \\ $
# 
# 

# In[1]:


# Se activan los truquitos:

import numpy as np
from sage.symbolic.integration.integral import definite_integral
import sys  


# In[70]:


# Se define el campo:

a,b,m0,m1=var('a,b,m1,m2')

F(x,y,z)=[a*(y - x - m1*x - ((m0-m1)/2)*(abs(x+1) - abs(x-1))) ,x-y+z , -b*y]
show(F)


# ## 1º

# #### a) Transformar el sistema en un sistema autonomo equivalente

# Puesto que $x',y',z'$ dependen unicamente de las variables $x,y,z$ sin depender explicitamente de $t$ el sistema proporcionado ya es autónomo.

# #### b) Calcular los puntos de equilibrio

# Los puntos de equilibrio de una ecuación diferencial se definen como los ceros del campo de dicha ecuación, en este caso basta buscar las soluciones del sistema de ecuaciones $ \\ $ $$\alpha(y-0.5x + \frac{1}{2}(|x+1|-|x-1|))=0, \quad x-y+z=0, \quad -\beta y=0 $$ $$$$ Se resuelve dicho sistema:

# In[71]:


m0=-3/2
m1=-1/2
F(x,y,z)=[a*(y - x - m1*x - ((m0-m1)/2)*(abs(x+1) - abs(x-1))) ,x-y+z , -b*y]

x0,x1,x2=solve([F(x,y,z)[0]==0,F(x,y,z)[1]==0,F(x,y,z)[2]==0],x,y,z)
x0=np.asarray(x0)
x1=np.asarray(x1)
x2=np.asarray(x2)

show(x0,x1,x2)


# #### c) Estudiar el linealizado de cada punto de equilibrio en el caso $\alpha=\beta=2$

# Los ceros del sistema de ecuaciones no dependen de los esclares $\alpha$ y $\beta$ puesto que estos estan multiplicando a toda una ecuación, por lo tanto los puntos fijos siguen siendo:

# In[92]:


a=b=2
m0=-3/2
m1=-1/2
F(x,y,z)=[a*(y - x - m1*x - ((m0-m1)/2)*(abs(x+1) - abs(x-1))) ,x-y+z , -b*y]

x0,x1,x2=solve([F(x,y,z)[0]==0,F(x,y,z)[1]==0,F(x,y,z)[2]==0],x,y,z)
x0=np.asarray(x0)
x1=np.asarray(x1)
x2=np.asarray(x2)
show(F)
show(x0,x1,x2)


# In[96]:


# Se define una función que nos permite evaluar la matriz jacobiana en un punto:

def evaluar_matriz_de_polinomios(matriz, valores):
    filas = len(matriz[0][:])
    columnas = len(matriz[:][0])
    resultado = [[None for _ in range(columnas)] for _ in range(filas)]
    valores= ({x : valores[0] , y : valores[1] , z : valores[2]})
    
    for i in range(filas):
        for j in range(columnas):
            polinomio = matriz[i][j]
            resultado[i][j] = polinomio.subs(valores)

    return matrix(CDF,resultado)


# In[74]:


# Se define una función que estudia la estabilidad de un punto crítico a partir de la matriz obtenida
# por la linealización en dicho punto.

def estudio_estabilidad(A):
    n=len(A[0])
    partes_reales = [numero.real_part() for numero in A.eigenvalues()]
    asin=0
    for i in range(n):
        if partes_reales[i]>0:
            print("el punto es inestable")
            break
        if partes_reales[i]<0:
            asin=asin+1
    if asin==3:
        print("el punto es asintoticamente estable")


# Para linealizar el sistema de ecuaciones se calculará la matriz jacobiana en cada uno de los puntos de equilibrio. Empezamos con x0:

# In[114]:


# Se representa el campo de vectores alrededor de x0, en un entorno de x0 el campo es lineal:

plot_vector_field3d(F(x,y,z),(x,-3,-1),(y,-1,1),(z,1,3))


# In[117]:


# Se calcula la matriz jacobiana y se evalua en el punto a traves de la función definida anteriormente: 

A=jacobian([F(x,y,z)[0],F(x,y,z)[1],F(x,y,z)[2]] , [x,y,z])

x0=vector([-2,0,2])

A_x0=evaluar_matriz_de_polinomios(A,x0)

show(A_x0)


# Se clasifica el punto crítico x0, para ello se calculan los autovalores de A_x0 y se estudia su parte real.
# Hacer notar que para poder realizar dicha clasificación se necesita que el sistema de ecuaciones tenga las primeras derivadas parciales continuas en un entorno de x0 , en F(x,y,z) caso las derivadas parciales son continuas salvo en x=1 ó x=-1.

# In[118]:


estudio_estabilidad(A_x0)


# Se realiza el mismo procedimiento con x1:

# In[116]:


# Se representa el campo de vectores alrededor de x1, en un entorno de x1 el campo es lineal:

plot_vector_field3d(F(x,y,z),(x,-1,1),(y,-1,1),(z,-1,1))


# In[11]:


# Se calcula la matriz jacobiana y se evalua en el punto a traves de la función definida anteriormente: 

A=jacobian([F(x,y,z)[0],F(x,y,z)[1],F(x,y,z)[2]] , [x,y,z])

x1=vector([0,0,0])

A_x1=evaluar_matriz_de_polinomios(A,x1)

show(A_x1)


# Se clasifica el punto crítico x1, para ello se calculan los autovalores de A_x1 y se estudia su parte real:

# In[12]:


estudio_estabilidad(A_x1)


# y con x2:

# In[102]:


# Se representa el campo de vectores alrededor de x2, en un entorno de x2 el campo es lineal:

plot_vector_field3d(F(x,y,z),(x,1,3),(y,-1,1),(z,-3,-1))


# In[103]:


# Se calcula la matriz jacobiana y se evalua en el punto a traves de la función definida anteriormente: 

A=jacobian([F(x,y,z)[0],F(x,y,z)[1],F(x,y,z)[2]] , [x,y,z])

x2=vector([2,0,-2])

A_x2=evaluar_matriz_de_polinomios(A,x2)

show(A_x2)


# Se clasifica el punto crítico x1, para ello se calculan los autovalores de A_x1 y se estudia su parte real:

# In[104]:


estudio_estabilidad(A_x2)


# ## 2º

# In[13]:


# Se toma como condición inicial (x(0),y(0),z(0))=(2,2,2) :

ini=vector(RDF,[2,2,2])


# Solo se pide uno de los dos métodos, pero también le aplico el método de Taylor de 2º orden al campo del sistema de ecuaciones diferenciales para ver si es una buena aproximación

# In[14]:


# Se define una función que aplica el método de Taylor al campo:

def taylor(F, ini , t, n):
    k = [0] * (n + 1)
    k[0] = ini
    h = (t - 0)/n  
    for i in range(n):
        xk, yk, zk = k[i]        
        f1x, f1y, f1z = F
        f2xx = f1x.diff(x)(xk, yk, zk)
        f2xy = f1x.diff(y)(xk, yk, zk)
        f2xz = f1x.diff(z)(xk, yk, zk)
        f2yy = f1y.diff(y)(xk, yk, zk)
        f2yz = f1y.diff(z)(xk, yk, zk)
        f2zz = f1z.diff(z)(xk, yk, zk)

        k[i+1] = [xk + (h)*f1x(xk, yk, zk).n() + ((h**2)/6)*(f2xx + 2*f2xy + 2*f2xz + f2yy + 2*f2yz + f2zz),
                  yk + (h)*f1y(xk, yk, zk).n() + ((h**2)/6)*(f2yy + 2*f2yz + f2zz),
                  zk + (h)*f1z(xk, yk, zk).n()]
    return k


# In[15]:


# Se define una función que aplica el método de Runge-Kutta:

def Runge_Kutta(F,ini,t,n):
    k = [0.] * (n + 1)
    k[0] = ini
    h = (t - 0)/n
    for i in range(n):
        xk, yk, zk = k[i]

        K1 = F(xk, yk, zk).n()
        K2 = F(xk + (h/2) * K1[0], yk + (h/2) * K1[1], zk + (h/2) * K1[2])
        K3 = F(xk + (h/2) * K2[0], yk + (h/2) * K2[1], zk + (h/2) * K2[2])
        K4 = F(xk + h * K3[0], yk + h * K3[1], zk + h * K3[2])

        x_nuevo = xk + (h/6) * (K1[0] + 2*K2[0] + 2*K3[0] + K4[0])
        y_nuevo = yk + (h/6) * (K1[1] + 2*K2[1] + 2*K3[1] + K4[1])
        z_nuevo = zk + (h/6) * (K1[2] + 2*K2[2] + 2*K3[2] + K4[2])

        k[i+1] = [x_nuevo, y_nuevo, z_nuevo]

    return k


# In[119]:


# Se aplican ambos métodos con 100 pasos, Se van a aproximar las soluciones en t=1, t=-1 y t=3:

sol_taylor_1 = taylor(F,ini,1,100)
sol_runge_1 = Runge_Kutta(F,ini,1,100)

print("Método de Taylor con 100 pasos para t=1:",sol_taylor_1[100])
print("Método de Runge-Kutta con 100 pasos para t=1:",sol_runge_1[100])

sol_taylor_menos_1 = taylor(F,ini,-1,100)
sol_runge_menos_1 = Runge_Kutta(F,ini,-1,100)

print("Método de Taylor con 100 pasos para t=-1:",sol_taylor_menos_1[100])
print("Método de Runge-Kutta con 100 pasos para t=-1:",sol_runge_menos_1[100])

sol_taylor_3 = taylor(F,ini,3,100)
sol_runge_3 = Runge_Kutta(F,ini,3,100)

print("Método de Taylor con 100 pasos para t=7:",sol_taylor_3[100])
print("Método de Runge-Kutta con 100 pasos para t=7:",sol_runge_3[100])


# In[120]:


# Se dibuja el método para t=1:

point(sol_taylor_1,color="red") + plot_vector_field3d(F(x,y,z),(x,-5,5),(y,-5,5),(z,-5,5))+point(sol_runge_1,color="blue") + plot_vector_field3d(F(x,y,z),(x,-5,5),(y,-5,5),(z,-5,5))


# In[121]:


# Se dibuja el método para t=-1:

point(sol_taylor_menos_1,color="red") + plot_vector_field3d(F(x,y,z),(x,-5,5),(y,-5,5),(z,-5,5))+point(sol_runge_menos_1,color="blue") + plot_vector_field3d(F(x,y,z),(x,-5,5),(y,-5,5),(z,-5,5))


# In[122]:


# Se dibuja el método para t=3:

point(sol_taylor_3,color="red") + plot_vector_field3d(F(x,y,z),(x,-5,5),(y,-5,5),(z,-5,5))+point(sol_runge_3,color="blue") + plot_vector_field3d(F(x,y,z),(x,-5,5),(y,-5,5),(z,-5,5))


# #### b)

# Aquí divido el apartado en 2 partes, una primera donde calculo el error de truncamiento del método y una segunda parte donde calculo el error del método de orden 2 con respecto al método de orden 4

# ##### b.1)

# primero hay que resolver la ecuación diferencial, el problema es que el campo no es diferenciable en todo punto, por lo tanto para obtener las soluciones se divide el campo, definiendolo unicamente en los intervalos de continuidad:
# 
# $F(x,y,z) =
# \begin{cases}
#   (2\cdot y(t) -  x(t) - 2 \, ,\, x(t) - y(t) + z(t) \, ,\, -2\cdot y(t)) & \text{si } x < -1 \\
#   (2\cdot y(t)  + x(t)  \,,\, x(t) - y(t) + z(t) \,, \, -2\cdot y(t)) & \text{si } -1 < x < 1 \\
#   (2\cdot y(t) -  x(t) + 2 \, ,\, x(t) - y(t) + z(t) \, ,\, -2\cdot y(t))) & \text{si } x > 1 \\
# \end{cases}$

# In[123]:


x,y,z=var('x,y,z')

f_mm1(x,y,z)=[2*y - x - 2, x - y - z, -2*y]
f_tt1(x,y,z)=[2*y + x, x - y - z, -2*y]
f_e1(x,y,z)=[2*y - x + 2, x - y - z, -2*y] 

# A través de un programa se resuelven las ecuaciones diferenciales:


# In[124]:


# Resolvemos para el caso x < -1:

t, C1m, C2m, Cm = var('t C1m C2m Cm')

eq1_mm1(t,C1m,C2m,Cm) = sqrt(5)*(C2m*exp(2*sqrt(5)*t) - C1m) - 5*C1m / exp(sqrt(5)*t) - 5*C2m*exp(sqrt(5)*t) - 2*exp(2*t) - 2/5
eq2_mm1(t,C1m,C2m,Cm) = -2*sqrt(5)*C2m*exp(sqrt(5)*t) + 2*sqrt(5)*C1m/exp(sqrt(5)*t) - 3*exp(2*t) + 4/5
eq3_mm1(t,C1m,C2m,Cm) =  4*C2m*exp(sqrt(5)*t) + 4*C1m/exp(sqrt(5)*t) + 3*exp(2*t) - 8*t/5 + Cm

system_of_equations = [eq1_mm1-ini[0], eq2_mm1-ini[1], eq3_mm1-ini[2]]

# Se obtienen los coeficientes para los valores iniciales

solutions_new = solve([eq.subs(t == 0) for eq in system_of_equations], C1m , C2m, Cm, solution_dict=True)

sol = solutions_new[0]

v=[sol[C1m].n(),sol[C2m].n(),sol[Cm].n()]

sol_mm1(t) = [eq1_mm1(t,*v), eq2_mm1(t,*v), eq3_mm1(t,*v)]
show(sol_mm1)


# In[125]:


# Resolvemos para el caso x > 1:

t, C1e, C2e, Ce= var('t C1e C2e Ce ')

eq1_e1(t,C1e,C2e,Ce) = (sqrt(5)*(C2e*exp(2*sqrt(5)*t) - C1e) - 5*C1e) / exp(sqrt(5)*t) - 5*C2e*exp(sqrt(5)*t) - 2*exp(2*t) + 2/5
eq2_e1(t,C1e,C2e,Ce) = -2*sqrt(5)*C2e*exp(sqrt(5)*t) + 2*sqrt(5)*C1e/exp(sqrt(5)*t) - 3*exp(2*t) - 4/5
eq3_e1(t,C1e,C2e,Ce) = 4*C2e*exp(sqrt(5)*t) + 4*C1e/exp(sqrt(5)*t) + 3*exp(2*t) + 8*t/5 + Ce

system_of_equations = [eq1_e1-ini[0], eq2_e1-ini[1], eq3_e1-ini[2]]

solutions = solve([eq.subs(t == 0)  for eq in system_of_equations], C1e, C2e, Ce, solution_dict=True)

sol = solutions[0]

v=[sol[C1e].n(),sol[C2e].n(),sol[Ce].n()]

sol_e1(t) = [eq1_e1(t,*v), eq2_e1(t,*v), eq3_e1(t,*v)]
show(sol_e1)


# La única solución para el caso -1< x(t) < 1 es x(t)=0, y(t)=0, z(t)=0 

# Se realiza un desarrollo de Taylor de 4º grado en cada una de las funciones:

# In[126]:


def der_total_sistemas(F,n):
    if n==0:
        return F
    dtn = der_total_sistemas(F,n-1)
    return  dtn.diff() * F


# In[127]:


def Taylor(T,a,b,ini,n):
    h = (b-a)/n
    xk = (n+1)*[ini]
    xk[0] = ini.n()
    for i in range(n):
        xk[i+1] = (xk[i] + h * T(*xk[i],h)).n()       
    return xk


# In[ ]:


# Si x0 > 1 aplicamos el desarollo de taylor de 4º orden a la función que define el campo en elintervalo x > 1:


# In[128]:


T4_e1(x,y,z,h) = der_total_sistemas(f_e1,0) +\
            der_total_sistemas(f_e1,1)*h/2 +\
            der_total_sistemas(f_e1,2)*h^2/6 +\
            der_total_sistemas(f_e1,3)*h^3/24
show(T4_e1)


# In[ ]:


# Si -1 < x0 < 1 aplicamos el desarollo de taylor de 4º orden a la función que define el campo en elintervalo -1 < x < 1:


# In[129]:


T4_tt1(x,y,z,h) = der_total_sistemas(f_tt1,0) +\
            der_total_sistemas(f_tt1,1)*(h/2) +\
            der_total_sistemas(f_tt1,2)*((h^2)/6) +\
            der_total_sistemas(f_tt1,3)*((h^3)/24)
show(T4_tt1)


# In[ ]:


# Si x0 < -1 aplicamos el desarollo de taylor de 4º orden a la función que define el campo en elintervalo x < -1:


# In[130]:


T4_mm1(x,y,z,h) = der_total_sistemas(f_mm1,0) +\
            der_total_sistemas(f_mm1,1)*(h/2) +\
            der_total_sistemas(f_mm1,2)*((h^2)/6) +\
            der_total_sistemas(f_mm1,3)*((h^3)/24)
show(T4_mm1)


# Una vez hechos los desarrollos en Taylor de 4º orden se procede a calcular el error de truncamiento

# In[131]:


# Todo este proceso se va a realizar unicamete sobre el valor inicial (x(0),y(0),z(0))=(2,2,2) y t=1:
ini=vector(RDF,[2,2,2])
a=0
b=1
n=100


# In[132]:


if ini[0] < -1:
    tru=vector([0.]*(n+1))
    for i in range(n):
        h=(b-a)/n
        re=sol_mm1(a+i*h).n()
        tru[i]=abs((re - sol_mm1(a+(i-1)*h).n())/h + h*T4_mm1(*re,h))  
    error_t=max(tru)
    print(error_t)
if -1 < ini[0] < 1:
    print('No es posible calcular el error de truncamiento')
    
if ini[0] > 1:
    tru=vector([0.]*(n+1))
    for i in range(n):
        h=(b-a)/n
        
        re=sol_e1(a+i*h).n()
        tru[i]=abs((re - sol_e1(a+(i-1)*h).n())/h + h*T4_e1(*re,h))  
    error_t=max(tru)
    print(error_t)

# Entiendo que la solución está mal, el error debe de estar en el la 
# determinación de los coeficientes para una solución concreta del sistema de ecuaciones diferenciales


# ##### b.2)

# Calculamos el error global del método de Taylor de orden 4 respecto al método de Taylor de orden 2

# In[134]:


# para t=1
a=0
b=1
ini=vector(RDF,[2,2,2])
s=1
T2_e1(x,y,z,h) = der_total_sistemas(f_e1,0) +\
            der_total_sistemas(f_e1,1)*(h/2) 

T2_mm1(x,y,z,h) = der_total_sistemas(f_mm1,0) +\
            der_total_sistemas(f_mm1,1)*(h/2) 

T2_tt1(x,y,z,h) = der_total_sistemas(f_tt1,0) +\
            der_total_sistemas(f_tt1,1)*(h/2) 

m=100
for i in range(0,10000,m):

    T_4_e1=Taylor(T4_e1,a,b,ini,s)
    T_4_mm1=Taylor(T4_mm1,a,b,ini,s)
    T_4_tt1=Taylor(T4_tt1,a,b,ini,s)

    T_2_e1=Taylor(T2_e1,a,b,ini,s)
    T_2_mm1=Taylor(T2_mm1,a,b,ini,s)
    T_2_tt1=Taylor(T2_tt1,a,b,ini,s)

    if ini[0] < -1:
        error_1=abs(T_4_mm1[s] - T_2_mm1[s])
    if -1 < ini[0] <1:
        error_1=abs(T_4_tt1[s] - T_2_tt1[s])
    if ini[0] > 1:
        error_1=abs(T_4_e1[s] - T_2_e1[s])  

# para acelerar la búsqueda de s:

    if error_1 < 10^(-5):
        s=s-m
        m=m/10
    if m == 1:
        break
    s+=m

print('El nº de pasos que se deben realizar para tener un error menor a 10^-5 es:',s,'el cual nos da un error de',error_1)


# #### c)

# Aplico el método de Adams-Bashford de 3 pasos tomando como valores iniciales 3 aproximaciones aportadas por el método de Runge-Kutta de un orden y un número de pasos predeterminados.

# In[135]:


# se vuelve a partir de la condición inicial (x(0),y(0),z(0))=(1,1,1) :

ini=vector([2,2,2])
a=0


# In[136]:


# Se pide una función F, tres valores iniciales, el tiempo final y el número de pasos del método

def Adams_Bashford(F,x0,x1,x2,t,a,n):
    kn=[x0]*(n+1)
    kn[0]=vector(x0)
    kn[1]=vector(x1)
    kn[2]=vector(x2)
    h=(t-a)/n
    for i in range(2,n):
        kn[i+1]=kn[i] + (h/12)*(23*F(*kn[i]) - 16*F(*kn[i-1]) + 5*F(*kn[i-2]))
        
    return kn


# In[137]:


# Se aproximan las soluciones para t=1, t=-1 y t=3 con 50 pasos
# partiendo de las 3 primeras aproximaciones del método de Runge-Kutta de 3º orden con de 100 pasos:


sol_Adams_Bashford_1=Adams_Bashford(F,sol_runge_1[0],sol_runge_1[1],sol_runge_1[2],2,a,50)

print("Método de Adams-Bashford con 3 pasos para t=1:",sol_Adams_Bashford_1[3])

sol_Adams_Bashford_menos_1=Adams_Bashford(F,sol_runge_menos_1[0] ,sol_runge_menos_1[1] ,sol_runge_menos_1[2],-1,a,50)


print("Método de Adams-Bashford con 3 pasos para t=-1:",sol_Adams_Bashford_menos_1[3])

sol_Adams_Bashford_3=Adams_Bashford(F,sol_runge_3[0] ,sol_runge_3[1] ,sol_runge_3[2],3,a,50)

print("Método de Adams-Bashford con 3 pasos para t=3:",sol_Adams_Bashford_3[3])


# #### d)

# In[138]:


def Adams_Corrector(F,x0,x1,x2,t,a,n):
    kn=[x0]*(n+1)
    kn_barra=[x0]*(n+1)
    kn[0]=vector(x0)
    kn[1]=vector(x1)
    kn[2]=vector(x2)
    h=(t-a)/n
    for i in range(2,n):
        kn_barra[i+1]=kn[i] + (h/12)*(23*F(*kn[i]) - 16*F(*kn[i-1]) + 5*F(*kn[i-2]))
        kn[i+1]=kn[i-1] + (h/12)*(5*F(*kn_barra[i+1]) + 8*F(*kn[i])) - F(*kn[i-1])
                 
    return kn,kn_barra


# In[139]:


#Se aproximan las soluciones para t=1, t=-1 y t=3 con 50 pasos partiendo de las 3 primeras aproximaciones
#del método de Runge-Kutta de 3º orden con 100 pasos:


sol_Adams_Corrector_1,sol_Adams_Corrector_1_barra=Adams_Corrector(F,sol_runge_1[0],sol_runge_1[1],sol_runge_1[2],1,a,50)

print("Método de Predictor-Corrector con 3 pasos para t=1:",sol_Adams_Bashford_1[3])

sol_Adams_Corrector_menos_1,sol_Adams_Corrector_menos_1_barra=Adams_Corrector(F,sol_runge_menos_1[0],sol_runge_menos_1[1],sol_runge_menos_1[2],-1,a,50)


print("Método de Predictor-Corrector con 3 pasos para t=-1:",sol_Adams_Corrector_menos_1[3])

sol_Adams_Corrector_3,sol_Adams_Corrector_3_barra=Adams_Corrector(F,sol_runge_3[0],sol_runge_3[1],sol_runge_3[2],3,a,50)

print("Método de Predictor-Corrector con 3 pasos para t=3:",sol_Adams_Corrector_3[3])


# In[140]:


# el error global se calcula de la siguiente manera:

e_global_1=abs((sol_Adams_Corrector_1_barra[3] - sol_Adams_Corrector_1[3])/10)

e_global_menos_1=abs((sol_Adams_Corrector_menos_1_barra[3] - sol_Adams_Corrector_menos_1[3])/10)

e_global_3=abs((sol_Adams_Corrector_3_barra[3] - sol_Adams_Corrector_3[3])/10)


print('El error global para t=1 es:',e_global_1)
print('El error global para t=-1 es:',e_global_menos_1)
print('El error global para t=3 es:',e_global_3)


# ## 3º

# Se estudia el linealizado general, para ello se toma un punto aleatorio el cual debe de introducir el usuario, se linealiza el sistema de ecuaciones diferenciales en dicho punto a traves de la matriz jacobiana y se obtiene una base de soluciones.

# In[141]:


A=jacobian([F(x,y,z)[0],F(x,y,z)[1],F(x,y,z)[2]] , [x,y,z])

valor=vector(RDF,[0]*3)
valor=[2,4,3]
if valor[0] != 1 and valor[0] != -1:
    B=evaluar_matriz_de_polinomios(A,valor)
else :
    print("El sistema de ecuaciones diferenciales no es diferenciable en ese punto, no se puede calcular una solución")
    sys.exit()
    
value=vector(CDF,B.eigenvalues()).n()

vect=B.eigenvectors_right()

vectores = [vect[i][1] for i in range(3)]
multi=[vect[i][2] for i in range(3)]

prueba = []
for lista1 in vectores:
    for numero in lista1:
        prueba.append(numero)

t=var('t')
for i in range(3):
    sol[i]=prueba[i]*e^((value[i])*t)
    
print("una base de soluciones del sistema de ecuaciones lineal en dicho punto es:")
print()
show(sol[0])
print()
show(sol[1])
print()
show(sol[2])


# In[ ]:





# ## Anexo

# Había intentado realizar los métodos de Adams de forma recursiva, como te comenté en clase, los dejo por aquí

# Adams-Bashford:

# In[ ]:


def Bashford(F,x_ini,k,t):
    nk = [vector(x_ini)] * (k + 1)
    nk[0]=x_ini
    h=(t-0)/k
    w=var("w")

    for i in range(k):
        xk, yk, zk = nk[i]
        sum=vector([0,0,0])
        for j in range(i):
            prod=1
            for u in range(i):
                if u!=j:
                    prod=prod*((w-u)/(j-u))
                b=definite_integral(prod,w,i,i+1)
                sum=sum+b*F(*nk[j])
        xk_nuevo = xk + h * sum[0]
        yk_nuevo = yk + h * sum[1]
        zk_nuevo = zk + h * sum[2]
        nk[i + 1] = vector([xk_nuevo, yk_nuevo, zk_nuevo])
    
    return nk


# In[ ]:


# Se aproximan las soluciones para t=2, t=5, t=7:

sol_Bashford_2=Bashford(F,ini,3,2)

print("Método de Adams-Bashford con 3 pasos para t=2:",sol_Bashford_2[3])

sol_Bashford_5=Bashford(F,ini,3,5)


print("Método de Adams-Bashford con 3 pasos para t=5:",sol_Bashford_5[3])

sol_Bashford_7=Bashford(F,ini,3,7)

print("Método de Adams-Bashford con 3 pasos para t=7:",sol_Bashford_7[3])


# Adams-Bashford/Adams_Moulton como Predictor/Corrector:

# Ahora se define el método corrector, este método puede resultar bastante lioso, voy a intentar explicar lo que se pretende realzizar:
# 
# Se parte de un único valor inicial denotado x0, se utiliza el método de Adams-Bashford de 1 paso como método predictor y se utiliza el método de Adams-Moulton de 1 paso como corrector, la aproximación del método corrector, que denotaremos como x1, junto con x0, nos permite aplicar el método de Adams-Bashford de 2 pasos como método predictor y el de dams-Moulton de 2 pasos como corrector, obteniendo otra aproximación, x2, esto se repite un número k de veces.

# In[142]:


def Corrector(F,x_ini,k,t):
    nk = [vector(x_ini)] * (k + 1)
    nk[0]=x_ini
    h=(t-0)/k
    w=var("w")
    xk= vector(RDF,[0]*(k+1))
    yk= vector(RDF,[0]*(k+1))
    zk= vector(RDF,[0]*(k+1))
    for i in range(k):
        xk[i], yk[i], zk[i] = nk[i]
        sum=vector([0,0,0])
        for j in range(i):
            prod=1
            for u in range(i):
                if u!=j:
                    prod=prod*((w-u)/(j-u))
                b=definite_integral(prod,w,i,i+1)
                sum=sum+b*F(xk[j],yk[j],zk[j])
        xk_barra = xk[i] + h * sum[0]
        yk_barra = yk[i] + h * sum[1]
        zk_barra = zk[i] + h * sum[2]
        for l in range(i+1):
            prod_barra=1
            if l!=i:
                prod_barra=prod_barra*((w-l)/(i-l))
        b=definite_integral(prod_barra,w,i,i+1)
        sum_barra=b*F(xk_barra,yk_barra,zk_barra)
        
        xk_nuevo = h*sum_barra[0] + xk_barra
        yk_nuevo = h*sum_barra[1] + yk_barra
        zk_nuevo = h*sum_barra[2] + zk_barra
        nk[i + 1] = vector([xk_nuevo, yk_nuevo, zk_nuevo])
    
    return nk


# In[143]:


# Se aplica el método corrector para t=2, t=5 y t=7:

sol_corrector_2=Corrector(F,ini,3,2)

print("Método de Corrector con 3 pasos para t=2:",sol_corrector_2[3])

sol_corrector_5=Corrector(F,ini,3,5)

print("Método de Corrector con 3 pasos para t=5:",sol_corrector_5[3])

sol_corrector_7=Corrector(F,ini,3,7)

print("Método de Corrector con 3 pasos para t=7:",sol_corrector_7[3])
