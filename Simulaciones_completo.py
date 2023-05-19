# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:50:40 2023

@author: diego
"""
import Module_grad_hess_estimation as M
import numpy as np
import matplotlib.pyplot as plt
import math as mt

# Vamos a realizar las simulaciones para los diferentes casos
# Definimos la S, p y que para la función que vamos a estudiar
# Definimos también el centro inicial de todas las formaciones



S = 1e-04*np.array([[-1, 0, 0],
                    [0, -2, 0],
                    [0, 0, -3]])

p = np.array([[0], [0], [0]],dtype=float)
q = np.array([0], dtype=float)
c0 = np.array([[50], [50], [50]], dtype=float)

#Definimos la posición del máximo para esta función
cf = np.array([[0], [0], [0]])



# Redefinimos sigma
#Para añadir el ruido a la función, bastaría con sumarle un término aleatorio dentro de la propia función

def sigma(r):
    # Vamos a utilizar S , p y q predefinidos
    return 10000*mt.exp(r.T@S@r+p.T@r+q)
  
#Incluimos las funciones de gradiente y Hessiano reales, pues esto nos permite comparar el estimado con lo reales
def grad_sigma(r):
    return 2*10000*mt.exp(r.T@S@r+p.T@r+q)*(S@r+p)

def Hess_sigma(r):
    return 10000*mt.exp(r.T@S@r+p.T@r+q)*(4*(S@r)@(S@r).T+2*S)
    


#%%
# =============================================================================
# Empezamos por el gradiente
# =============================================================================

# Hagamos el algoritmo del gradiente de manera exacta y no exacta
# El gradiente exacto viene dado por:

#Hacemos el algoritmo para varios radios, pues eso nos permite comparar 
n=4
Ds = [0.1,1,10,100]
alpha = 0.1 
errorsT = []
maxit =250



for D in Ds:
    c = np.copy(c0)

    it = 0
    pos = M.formacion_grad3D(c, n, D)
    errors = np.empty(maxit)
    
    #Aquí aplicamos el algoritmo en sí
    while it < maxit:
        grad = M.estim_grad3D(sigma, pos, c, n, D)
        c += alpha*grad
        err = np.linalg.norm(c-cf, 1)
        errors[it] = err
        pos = M.formacion_grad3D(c, n, D)
        it += 1
    errorsT.append(errors)
c_gradreal = np.copy(c0)
errorsgradreal = np.empty(maxit)
it = 0

# Cálculo del gradiente real
while it < maxit:
    gradreal = grad_sigma(c_gradreal)
    c_gradreal += alpha*gradreal
    errorsgradreal[it] = np.linalg.norm(c_gradreal-cf, 1)
    it += 1

#Hacemos una figura para la comparación
itera = np.arange(0, maxit)
color = ['blue', 'black', 'purple', 'green']
style = ['-', '-', '-', '-']
fig, ax = plt.subplots()
for i in range(len(errorsT)):
    plt.plot(itera, errorsT[i], color[i],label='D='+str(Ds[i]), linestyle=style[i])
    plt.yscale("log")
    plt.xlabel('Iteraciones')
    plt.ylabel(r'Error $||c-r^*||$')
    
    plt.grid('on')


plt.plot(itera, errorsgradreal, 'r', linewidth=4, label='Gradiente exacto')
plt.legend()
plt.show()







# %%

#Aquí añadimos el error final tras un número suficiente de iteraciones para diferentes radios.
#Esto no se muestra en el paper, pero es interesante ver cómo la forma del error depende de la función
n = 4
maxit = 250
D = np.linspace(1, 50, 100)
errfinal = []

for d in D:
    it = 0
    c = np.copy(c0)
    pos = M.formacion_grad3D(c, n, d)
    while it < maxit:
        grad = M.estim_grad3D(sigma, pos, c, n, d)
        c += alpha*grad
        err = np.linalg.norm(c-cf, 1)
        pos = M.formacion_grad3D(c, n, d)
        it += 1
    err = np.linalg.norm(c-cf, 1)
    errfinal.append(err)
    # Vamos a calcular la pendiente

plt.plot(D, errfinal)
plt.xlabel('Radio de la formación (D)')
plt.ylabel(r'Error $||c-r^*||$')
plt.grid('on')
plt.show()

#Al error respecto al gradiente le pasa algo similar al de la posición
errorgrad=[]
for d in D:
    pos=M.formacion_grad3D(c, n, d)
    gradest=M.estim_grad3D(sigma, pos, c, n, d)
    gradreal=grad_sigma(c)
    errorgrad.append(np.linalg.norm(gradest-gradreal))
plt.plot(D,errorgrad)
plt.show()







# %%
# =============================================================================
# Vamos ahora con el método BFGS
# =============================================================================
# De nuevo podemos hacerlo usando el gradiente de manera exacta y no exacta
n = 4
Ds = [0.1, 1, 10,100]
errorsT = []
maxit = 1000
H_inv0 = -np.identity(3)  
dim = 3
tol=1e-16
its=[]


for D in Ds:
    cprev = np.copy(c0)
    pos = M.formacion_grad3D(cprev, n, D)
    errors = np.zeros(maxit)
    H_inv = np.copy(H_inv0)
    # Calculamos el primer gradiente fuera del while 
    grad0 = M.estim_grad3D(sigma, pos, cprev, n, D)
    cnext = cprev-alpha*H_inv@grad0
    difprod=1.
    it=0
    #Cortamos cuando la diferencia sea muy pequeña para no romper el algoritmo
    while it<maxit and tol<difprod:
        grad1 = M.estim_grad3D(sigma, pos, cnext, n, D)
        modgrad=np.linalg.norm(grad1)
        dif=np.linalg.norm(grad0-grad1)
        H_inv = M.calculo_estH(grad0, grad1, cprev, cnext, H_inv, dim)
        dhi = np.linalg.det(H_inv)
        if (dhi >= 0.) :
            H_inv = -1.*np.eye(3,3)
        cprev = np.copy(cnext)
        cnext -= alpha*H_inv@grad1
        difprod=np.abs(np.dot((cprev-cnext).T,grad1-grad0))
        
        err = np.linalg.norm(cnext-cf)
        errors[it] = err
        pos = M.formacion_grad3D(cnext, n, D)
        it += 1
        grad0 = np.copy(grad1)
    its.append(it)
    errorsT.append(errors)



#Incluimos el cálculo con el gradiente real 

cprev = np.copy(c0)
pos = M.formacion_grad3D(cprev, n, D)
errorreal = np.zeros(maxit)
H_inv = np.copy(H_inv0)
# Calculamos el primer gradiente fuera del while
grad0 = grad_sigma(cprev)
cnext = cprev-alpha*H_inv@grad0 

it=0
while it<maxit:
    grad1 =grad_sigma(c)
    H_inv = M.calculo_estH(grad0, grad1, cprev, cnext, H_inv, dim)
    dhi = np.linalg.det(H_inv)
    #Aplicamos esta condición para que el hessiano sea definido negativo siempre
    if (dhi >= 0.) :
        H_inv = -1.*np.eye(3,3)
    cprev = np.copy(cnext)
    cnext -= alpha*H_inv@grad1
    difprod=np.abs(np.dot((cprev-cnext).T,grad1-grad0))
    
    err = np.linalg.norm(cnext-cf)
    errorreal[it] = err
    it += 1
    grad0 = np.copy(grad1)
itreal=it
    
color = ['blue', 'black', 'purple', 'green']
style = ['-', '-', '-', '-']
fig, ax = plt.subplots()
for i in range(len(errorsT)):

    plt.plot(np.arange(its[i]), errorsT[i][0:its[i]], color[i], label='D='+str(Ds[i]), linestyle=style[i])
    plt.yscale("log")
    plt.xlabel('Iteraciones')
    plt.ylabel(r'Error $||c-r^*||$')
    plt.ylim(1e-16, 10000)
    plt.grid('on')
    
plt.plot(np.arange(maxit),errorreal,color='red', linewidth=2)
plt.legend()  
plt.show()



# %%
# =============================================================================
# Hacemos el método de Newton
# =============================================================================
n=4
#Cambiamos el alpha, pues en este algoritmo sí podemos tomar un alpha más grande
alpha=0.8
Ds = [0.1, 1, 10,100]
errorsT = []
maxit = 3000
c0=np.array([[20.],
                           [20.],
                           [20.]])


for D in Ds:
    c = np.copy(c0)
    it = 0
    pos = M.formacion_H3D(c, n, D)
    errors = np.empty(maxit)
    while it < maxit:
        grad = M.estim_grad3D(sigma, pos, c, n, D)
        grad_real=grad_sigma(c0)
        
        print(grad,grad_real)
        K = M.calculo_K3D(n, sigma, pos, D, c)
        H = M.calculo_H3D(K, n)
        
        if np.linalg.det(H)>-0.1:
            H = -np.eye(3,3)
        
        c -= alpha*np.linalg.inv(H)@grad
        err = np.linalg.norm(c-cf)
        errors[it] = err
        pos = M.formacion_H3D(c, n, D)
        it += 1
    
    errorsT.append(errors)
    
#Calculemos también con el Hessiano exacto
it=0
errorsHessreal=np.zeros(maxit)
c=c0
while it<maxit:
    grad=grad_sigma(c)
    Hess=Hess_sigma(c)

    if np.linalg.det(Hess)>-0.1:
        Hess = -np.eye(3,3)
    Hessinv=np.linalg.inv(Hess)
    c-=alpha*Hessinv@grad
    errorsHessreal[it]=np.linalg.norm(c-cf)
    it+=1

itera = np.arange(0, maxit)
color = ['blue', 'black', 'purple', 'green']
style = ['-', '-', '-', '-']


fig, ax = plt.subplots()
for i in range(len(errorsT)):
    print()
    plt.plot(itera, errorsT[i], color[i], label='D='+str(Ds[i]), linestyle=style[i])
    
    plt.yscale("log")
    plt.xlabel('Iteraciones')
    plt.ylabel(r'Error $||c-r^*||$')
    plt.grid('on')
    
plt.plot(itera, errorsHessreal,'aqua',linewidth=3, label='Hessiano exacto')

plt.legend()
plt.show()

#%%
#Veamos ahora el error respecto a D como hicimos con el gradiente
n = 4
maxit = 250
D = np.linspace(1, 50, 100)
errfinal = []
for d in D:
    it = 0
    c = np.copy(c0)
    pos = M.formacion_H3D(c, n, d)
    while it < maxit:
            grad = M.estim_grad3D(sigma, pos, c, n, d)
            K = M.calculo_K3D(n, sigma, pos, d, c)
            H = M.calculo_H3D(K, n)
            if np.linalg.det(H)>-1:
                H = -np.eye(3,3)
            c -= alpha*np.linalg.inv(H)@grad
            pos = M.formacion_H3D(c, n, d)
            it += 1
    err = np.linalg.norm(c-cf, 1)
    errfinal.append(err)
#%%
#Hacemos el plot
fig, ax = plt.subplots() 
plt.plot(D,errfinal,'purple')
plt.xlabel('Radio de la formación (D)')
plt.ylabel(r'Error $||c-r^*||$')
plt.grid('on')
plt.show()






