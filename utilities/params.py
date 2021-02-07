
def getParamGeneral():
    alpha = 0.0005  # reconstruction loss coeff
    n_epochs = 100

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    init_sigma = 0.1
    caps1_n_dims = 8
    caps2_n_dims = 16
    caps1_n_maps = 32

    primary_cap_size1 = 6
    primary_cap_size2 = 6

    n_hidden1 = 512
    n_hidden2 = 1024

    return alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2


def getParamCaps(dsname):

    alpha, n_epochs, m_plus, m_minus, lambda_, \
    init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
    primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2 = getParamGeneral()
    
    if dsname == 'Cedar':
        num_class = 55
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/CEDAR_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.02

    if dsname == 'MCYT75':
        num_class = 75
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/MCYT75_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.05

    return num_class, image_size1, image_size2, num_image_channel, \
           checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2


def getBatchSize(dsname):
    batchSize = 100
    
    if dsname == 'MCYT75':
        batchSize = 2
    if dsname == 'cedar':
        batchSize = 5
    
    return batchSize

