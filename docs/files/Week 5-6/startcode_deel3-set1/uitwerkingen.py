import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.ops.confusion_matrix import confusion_matrix


def load_model():
    # Deze methode laadt het getrainde model dat je bij de vorige opgavenset heb
    # opgeslagen. 

    return keras.models.load_model('fashion.keras')

# OPGAVE 1a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix

    return confusion_matrix(labels, pred)

# OPGAVE 1b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html

    list = []

    for i in range(len(labels)):
        tp = conf[i][i]
        fp = np.sum(conf[:,i]) - tp
        fn = np.sum(conf[i,:]) - tp
        tn = np.sum(conf) - tp - fp - fn

        list.append((labels[i], tp, fp, fn, tn))

    return list


# OPGAVE 1c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    tp = sum([x[1] for x in metrics])
    fp = sum([x[2] for x in metrics])
    fn = sum([x[3] for x in metrics])
    tn = sum([x[4] for x in metrics])

    # BEREKEN HIERONDER DE JUISTE METRIEKEN EN RETOURNEER DIE 
    # ALS EEN DICTIONARY

    tpr = tp / (tp + fn)
    ppv = tp / (tp + tn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    rv = {'tpr':tpr, 'ppv':ppv, 'tnr':tnr, 'fpr':fpr }
    return rv