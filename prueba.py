""" import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
#Importa el DataSet
df = pd.read_csv('helados.csv', header=0, usecols=['consumo', 'ingreso', 'precio', 'temperatura'])
print(df.head())

#ajustar regresiones lineales entre consumo y las demás variables
fig, ax = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True, sharey=True)
ax[0].set_ylabel(df.columns[0])
for ax_, col in zip(ax, df.columns[1:]):
    res = scipy.stats.linregress(df[col], df["consumo"])
    x_plot = np.linspace(np.amin(df[col]), np. amax(df[col]), num=100)
    ax_.scatter(df[col], df["consumo"], label='datos', s=10)    
    ax_.plot(x_plot, res.slope*x_plot + res.intercept, lw=2, c='r', label='modelo')
    ax_.set_xlabel(col)
    ax_.set_title(f"$r$: {res.rvalue:0.5f}")
ax_.legend()
plt.show()

#verificar si las correlaciones son estadísticamente significativas
alfa = 0.05

for i, col in enumerate(df.columns[1:]):
    res = scipy.stats.linregress(df[col], df["consumo"])
    print(f"{col}: \t Rechazo hipótesis nula: {res.pvalue < alfa}")

#las distribuciones bajo la hipótesis nula: linea azul
#los límites dados por alfa: linea punteada negra (dos colas)
#El valor del observado para cada una de las variables: linea roja
fig, ax = plt.subplots(1, 3, figsize=(8, 2), tight_layout=True, sharey=True)
ax[0].set_ylabel(df.columns[0])

N = df.shape[0]
t = np.linspace(-7, 7, num=1000)
dist = scipy.stats.t(loc=0, scale=1, df=N-2) # dos grados de libertad


for i, col in enumerate(df.columns[1:]):
    res = scipy.stats.linregress(df[col], df["consumo"])
    t_data = res.rvalue*np.sqrt(N-2)/np.sqrt(1.-res.rvalue**2)
    ax[i].plot(t, dist.pdf(t))
    ax[i].plot([dist.ppf(alfa/2)]*2, [0, np.amax(dist.pdf(t))], 'k--')
    ax[i].plot([dist.ppf(1-alfa/2)]*2, [0, np.amax(dist.pdf(t))], 'k--')
    ax[i].plot([t_data]*2, [0, np.amax(dist.pdf(t))], 'r-')
    ax[i].set_xlabel(col)

plt.show() """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

ruta = input("Ingrese la ruta del archivo: ")
#Importa el DataSet
df = pd.read_csv(ruta, header=0, usecols=['Y', 'X1', 'X2'])
print(df.head())

#ajustar regresiones lineales entre consumo y las demás variables
fig, ax = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True, sharey=True)
ax[0].set_ylabel(df.columns[0])
for ax_, col in zip(ax, df.columns[1:]):
    res = scipy.stats.linregress(df[col], df["Y"])
    x_plot = np.linspace(np.amin(df[col]), np. amax(df[col]), num=100)
    ax_.scatter(df[col], df["Y"], label='datos', s=10)    
    ax_.plot(x_plot, res.slope*x_plot + res.intercept, lw=2, c='r', label='modelo')
    ax_.set_xlabel(col)
    ax_.set_title(f"$r$: {res.rvalue:0.5f}")
ax_.legend()
plt.show()

#verificar si las correlaciones son estadísticamente significativas
alfa = 0.05

for i, col in enumerate(df.columns[1:]):
    res = scipy.stats.linregress(df[col], df["Y"])
    print(f"{col}: \t Rechazo hipótesis nula: {res.pvalue < alfa}")

#las distribuciones bajo la hipótesis nula: linea azul
#los límites dados por alfa: linea punteada negra (dos colas)
#El valor del observado para cada una de las variables: linea roja
fig, ax = plt.subplots(1, 2, figsize=(8, 2), tight_layout=True, sharey=True)
ax[0].set_ylabel(df.columns[0])

N = df.shape[0]
t = np.linspace(-7, 7, num=1000)
dist = scipy.stats.t(loc=0, scale=1, df=N-2) # dos grados de libertad


for i, col in enumerate(df.columns[1:]):
    res = scipy.stats.linregress(df[col], df["Y"])
    t_data = res.rvalue*np.sqrt(N-2)/np.sqrt(1.-res.rvalue**2)
    ax[i].plot(t, dist.pdf(t))
    ax[i].plot([dist.ppf(alfa/2)]*2, [0, np.amax(dist.pdf(t))], 'k--')
    ax[i].plot([dist.ppf(1-alfa/2)]*2, [0, np.amax(dist.pdf(t))], 'k--')
    ax[i].plot([t_data]*2, [0, np.amax(dist.pdf(t))], 'r-')
    ax[i].set_xlabel(col)

plt.show()