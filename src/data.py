import numpy as np
from scipy.stats import norm


def transform_labels_gaussian(labels, t, freq):
    dummy = np.arange(-3, 3.001, 0.001)
    pdf_3 = norm.pdf(dummy) 

    labels_pdf = []
    for i, l in enumerate(labels):
        labels_pdf.append(list(np.zeros(max(0, -1*round(min(t)//freq)))) + 
                  list(np.zeros(round(l[0]/freq))) + 
                  list(np.interp(np.arange(round(l[0]/freq), round(l[1]/freq)+1, 1), 
                                 np.linspace(round(l[0]/freq), round(l[1]/freq), len(pdf_3)), 
                                 pdf_3)) + 
                  list(np.zeros(round(max(t)/freq)-round(l[1]/freq))))

    labels = np.array(labels_pdf)
    labels = np.reshape(labels, tuple([s for s in labels.shape]+[1]))

    return labels

def transform_labels_orig(labels, data, t, freq):
    labels_pdf = []
    for i, (l, d) in enumerate(zip(labels, data)):
        subset = d[round(l[0]/freq)+20:min(round(l[1]/freq)+21, 240)].flatten()
        max_value = np.max(np.abs(subset))
        labels_pdf.append(list(np.zeros(max(0, -1*round(min(t)//freq)))) + 
                  list(np.zeros(round(l[0]/freq))) + 
                  list(subset/max_value) +
                  list(np.zeros(round(max(t)/freq)-round(l[1]/freq))))

    labels = np.array(labels_pdf)
    labels = np.reshape(labels, tuple([s for s in labels.shape]+[1]))

    return labels

def transform_labels_binary(labels, t, freq):
    labels_pdf = []
    for i, l in enumerate(labels):
        labels_pdf.append(list(np.zeros(max(0, -1*round(min(t)//freq)))) + 
                  list(np.zeros(round(l[0]/freq))) + 
                  list(np.ones(round(l[1]/freq) - round(l[0]/freq) + 1)) + 
                  list(np.zeros(round(max(t)/freq)-round(l[1]/freq))))

    labels = np.array(labels_pdf)
    labels = np.reshape(labels, tuple([s for s in labels.shape]+[1]))

    return labels
