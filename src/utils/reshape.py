def flatten(w,b):
    w_reshape=np.asarray(w).flatten()
    b_reshape=np.asarray(w).flatten()
    theta=np.concat(w,b)
    return theta