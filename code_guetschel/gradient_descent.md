# Gradient descent

Une descente de gradient a pour but de minimiser une fonction de
coût $J(\theta)$ en modifiant les paramètres $\theta \in \mathbb{R}^d$
dans la direction oposée au gradient de la fonction de coût
 $\nabla_\theta J(\theta)$.

## Batch
Le gradient est calculé sur l'ensemble du dataset
puis les poids sont mis à jour selon la formule :
$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta)$$
Avec $\eta$ le pas d'apprentissage (learning rate).

## SGD
La descente de gradient stochastique consiste à
calculer le gradient et mettre à jour les paramètres
après le passage de chaque exemple d'apprentissage.
La même formule que en batch est utilisée.

## NAG
Nesterov accelerated gradient ajoute un terme d'inertie $\gamma v_{t-1}$
à la formule d mise à jour des paramètres.
Aussi NAG estime la prochaine position
des paramètres par $\theta-\gamma v_{t-1}$ et calculent le gradient de la fonction de coût par rapport à cette estimation.
Ce qui donne la formule de mise à jour suivante :
$$
\begin{split}
v_t &= \gamma v_{t-1} + \eta\nabla_\theta J( \theta - \gamma v_{t-1} ) \\
\theta &= \theta - v_t
\end{split}
$$
On prends généralement un terme d'inertie $\gamma$ à $0.9$

## Adagrad
Adagrad associe à chacun des paramètres $\theta_i$ un learning rate différent
(au lieu d'utiliser le même pour tous les paramètres).
Ces learning rates sont adaptatifs et décroissent.
Ceux associés à des paramètres qui évoluent beaucoup (forts gradients)
diminuent rapidement
tandis que les learning rates des paramètres qui évoluent lentement
(faibles gradients) diminuent lentement.
La mise à jour se fait ainsi :
$$
\begin{split}
g_{t, i} &= \nabla_\theta J( \theta_{t, i} )\\
G_{t, ii} &= \sum_{t'\le t} {g_{t',i}}^2 \\
\theta_{t+1, i} &= \theta_{t, i} - \dfrac{\eta}{\sqrt{G_{t, ii} + \epsilon}} \cdot g_{t, i}
\end{split}
$$
Avec $G_{t} \in \mathbb{R}^{d \times d}$ matrice diagonale, on peut vectoriser :
$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}
$$

## Adadelta
Adadelta est une extension de Adagrad.
Elle pare la décroissance stricte des learning rates de Adagrad
en remplacant la somme des gradients par une moyenne à lissage exponentiel.
Pour des raisons d'homogénéité, les auteurs remplacent le coefficient $\eta$
par une autre moyenne à lissage exponentiel mais cette fois-ci
sur les mises à jour des poids $\Delta\theta$.  

Formules de lissage exponentiel et de root mean squared error (`RMS`) :
$$
\begin{align}
E[X]_t &= \gamma E[X]_{t-1} + (1 - \gamma) X_t \\
RMS[X]_{t} &= \sqrt{E[X]_t + \epsilon}\\
\end{align}
$$

Formules de mise à jour des poids :
$$
\begin{align}
\Delta \theta_t &= - \dfrac{RMS[\Delta \theta]_{t-1}}{RMS[g]_{t}} g_{t} \\
\theta_{t+1} &= \theta_t + \Delta \theta_t
\end{align}
$$

## RMSprop
RMSprop a été développé en parallèle et pour les mêmes raisons que Adadelta.
Les deux méthodes sont relativement similaires mais RMSprop
ne s'occupe pas des questions d'homogénéité.
Voici la formules de mise à jour :
$$
\theta_{t+1} = \theta_t - \dfrac{\eta}{RMS[g]_{t}} g_{t}
$$
Les auteurs suggèrent $\gamma$ à $0.9$ et $\eta$ à $0.001$.

## Adam
Adaptive Moment Estimation (Adam) calcule aussi un learning rate adaptatif pour chaque paramètre.
En plus des carrés des gradients $v_t$ (comme Adadelta et RMSprop),
Adam maintiens un terme d'inertie $m_t$. Ils sont calculés comme cela :
$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{align}
$$

$m_t$ et $v_t$ sont initialisés à $0$.
Les auteurs observent que leurs valeurs sont biaisés vers $0$ lors des premières itérations.
Ils les corrigent donc comme ceci :
$$
\begin{align}
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2}
\end{align}
$$

Et les poids mis à jour comme ceci :
$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

Les auteurs proposent des valeurs par default suivantes :
$\beta_1=0.9$, $\beta_2=0.999$ et $\epsilon=10^{-8}$.
