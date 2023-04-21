def save_p(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
def open_p(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
# mesh of points between 0 and 2pi
theta_mesh = np.linspace(0, np.pi, 200)
phi_mesh = np.linspace(0, 2*np.pi, 200)

lower, upper = 2, 3
mu, sigma = 2.5, 0.1
S1 = stats.norm(loc=mu, scale=sigma)

lower, upper = 3.2, 4
mu, sigma = 3.5, 0.1
S2 = stats.norm(loc=mu, scale=sigma)

lower, upper = 4.2, 6
mu, sigma = 5, 0.1
S3 = stats.norm(loc=mu, scale=sigma)

def distro_distance(S):
    return S.rvs(1)

def from_xyz(xyz, axis=-1):
    x, y, z = np.moveaxis(xyz, axis, 0)

    lea = np.empty_like(xyz)

    pre_selector = ((slice(None),) * lea.ndim)[:axis]

    xy_sq = x ** 2 + y ** 2
    lea[(*pre_selector, 0)] = np.sqrt(xy_sq + z ** 2)
    lea[(*pre_selector, 1)] = np.arctan2(np.sqrt(xy_sq), z)
    lea[(*pre_selector, 2)] = np.arctan2(y, x)

    return lea


def to_xyz(lea, axis=-1):
    l, e, a = np.moveaxis(lea, axis, 0)

    xyz = np.empty_like(lea)

    pre_selector = ((slice(None),) * xyz.ndim)[:axis]

    xyz[(*pre_selector, 0)] = l * np.sin(e) * np.cos(a)
    xyz[(*pre_selector, 1)] = l * np.sin(e) * np.sin(a)
    xyz[(*pre_selector, 2)] = l * np.cos(e)

    return xyz


# define a function to get a random float between 0 and np.pi
def get_random_theta():
    return np.random.uniform(0,np.pi)

# define a function to get a random point between 0 and 360
def get_random_phi():
    return np.random.uniform(0,2*np.pi)

def get_random_r():
    return np.random.uniform(1,6)

def d_sphere(rtp1, rtp2):
    r1, th1, ph1 = rtp1
    r2, th2, ph2 = rtp2
    return np.sqrt(r1**2+r2**2-2*r1*r2*np.cos((th1-th2))-2*r1*r2*np.sin(th1)*np.sin(th2)*(np.cos((ph1-ph2))-1))

def sph_dis_matrix(list_of_points):
    dm = np.zeros((len(list_of_points),len(list_of_points)))
    for i in range(len(list_of_points)):
        for j in range(len(list_of_points)):
            dm[i,j]=d_sphere(list_of_points[i], list_of_points[j])
    return dm

def d_util_da(v):
    global good_points, r_target, d_target
    global log
    th2, ph2 = v
    if len(good_points)>2:
        check_points = np.concatenate((good_points, np.array([[r_target, th2, ph2]])))
        d0 = find_min(check_points)
    else:
        r1 = good_points[-1][0]
        th1 = good_points[-1][1]
        ph1 = good_points[-1][2]
        r2 = r_target
        d0=np.sqrt(r1**2+r2**2-2*r1*r2*np.cos((th1-th2))-2*r1*r2*np.sin(th1)*np.sin(th2)*(np.cos((ph1-ph2))-1))
        

    if d0 < d_target:
        return d0+d_target
    else:
        return d0

def find_min(list_of_points):
    tri = sph_dis_matrix(list_of_points)
    #tri = sph_dis_matrix(list_of_points[1:])
    tri_no_zeros = tri[tri!=0]
    
    return min(tri_no_zeros)

def find_a_point(radial, dis):
    global good_points, r_target, d_target
    r_target = radial
    d_target = dis
    result = dual_annealing(d_util_da, bounds)
    th2, ph2 = result.x
    return np.array([r_target, th2, ph2])

bounds = [[0, np.pi], [0, 2*np.pi]]


def get_rdf_abs(coords, abs_el1, rmesh):
    dm = fastdist.matrix_pairwise_distance(coords, fastdist.euclidean, "euclidean", return_matrix=True)
    mean = np.mean(coords, axis=0)
    all_ = fastdist.vector_to_matrix_distance(mean, coords, fastdist.euclidean, "euclidean")
    abs_el1 = dm[abs_el1]
    counts_abs_el1 = bin_list_mono(abs_el1, rmesh)
    return np.asarray(counts_abs_el1)


def normal_distro_S1(mean, std):
    global S1
    lower, upper = mean-.5, mean+.5
    mu, sigma = mean, std
    S1 = stats.norm(loc=mu, scale=sigma)

def normal_distro_S2(mean, std):
    global S2
    lower, upper = mean-.5, mean+.5
    mu, sigma = mean, std
    S2 = stats.norm(loc=mu, scale=sigma)

def normal_distro_S3(mean, std):
    global S3
    lower, upper = 4,6
    mu, sigma = mean, std
    S3 = stats.norm(loc=mu, scale=sigma)


def read_exafs_dat(file_path):
    return np.asarray([[float(s) for s in l.split()] for l in [l for l in read_lines(file_path) if l.split()[0] != '#']])[:,[2,5]]