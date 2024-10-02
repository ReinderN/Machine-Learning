import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    nrMatrix = np.reshape(nrVector, (20, 20), order='F')

    plt.matshow(nrMatrix, cmap=plt.cm.binary)
    plt.show()

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m × x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0, ... 0] zijn en als
    # y_i=10, dan is regel i in de matrix [0,0, ... 1] (10 symboliseert in deze dataset dus 9 en niet 0!).
    # In dit geval is de breedte van de matrix 10 (0-9),
    # maar de methode moet werken voor elke waarde van y en m

    x = np.max(y)
    cols = y.flatten() - 1
    rows = np.arange(m)
    data = np.ones(m, dtype=int)
    y_matrix = csr_matrix((data, (rows, cols)), shape=(m, x))
    return y_matrix

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta2, Theta3, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta2 en Theta3. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta2 en Theta3 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    m, _ = X.shape

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    a1 = np.hstack((np.ones((m, 1)), X))

    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    a2 = sigmoid(a1.dot(Theta2.T))

    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    a2 = np.hstack((np.ones((m, 1)), a2))

    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.
    a3 = sigmoid(a2.dot(Theta3.T))

    return a3

# ===== deel 2: =====
def compute_cost(Theta2, Theta3, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta2 en Theta3) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een ijle matrix.

    m, _ = X.shape

    h = predict_number(Theta2, Theta3, X)

    y_matrix = get_y_matrix(y, m).toarray()

    term1 = -y_matrix * np.log(h)
    term2 = (1 - y_matrix) * np.log(1 - h)

    cost = np.sum(term1 - term2) / m

    return cost


# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Controleer dat deze werkt met
    # scalaire waarden en met vectoren.

    return sigmoid(z) * (1 - sigmoid(z))

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta2, Theta3, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta2.shape)
    Delta3 = np.zeros(Theta3.shape)
    m, _ = X.shape

    y_matrix = get_y_matrix(y, m).toarray()

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta2.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2)))
    z3 = a2.dot(Theta3.T)
    a3 = sigmoid(z3)

    for i in range(m):
        error3 = a3[i] - y_matrix[i]
        error2 = Theta3.T.dot(error3) * sigmoid_gradient(np.hstack((1, z2[i])))

        Delta3 += np.outer(error3, a2[i])
        Delta2 += np.outer(error2[1:], a1[i])

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m
    
    return Delta2_grad, Delta3_grad