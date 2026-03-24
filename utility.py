import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from IPython.display import clear_output
from skimage.transform import resize
from skimage.data import chelsea
from skimage.restoration import estimate_sigma
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import subprocess
import sys
import os
import math

repo_url = "https://github.com/DmitryUlyanov/deep-image-prior.git"
repo_folder = "deep-image-prior"
if not os.path.exists(repo_folder):
    subprocess.run(["git", "clone", repo_url], check=True)
sys.path.append(os.path.abspath(repo_folder))
from models.skip import skip

title_fs = 24
label_fs = 18
tick_fs = 14

def torch_to_np(torch_array):
  """Recives a torch tensor and returns a numpy n-array object"""
  return np.squeeze( (255. * torch_array).byte().detach().cpu().numpy() )
def nrmse_f(recon, target):
    """Computates the mse of two n-arrays normalized to the target absolute mean """
    n = (recon - target)**2.
    d = target**2.
    return 100. * torch.mean(n)**0.5 / torch.mean(d)**0.5
def coefficient_reader(filename):
    """Recives a file path to a .txt using , for decimal notation and separated by spaces """
    x = np.loadtxt(filename, dtype=str)
    x = np.char.replace(x,",",".").astype(float)
    return x
def weighted_hadamard_from_vector(vec, k):
    """
    Recives:
        vec : a vector of size (2n,)
        k   : the size of hadamard matrices(k×k)

    Process:
        1. Switches the signs of every 2 index in one array
        2. Computates v_sum[i] = vec[2*i] + vec[2*i + 1].
        3. Generates Hadamard matrices in jpeg algorithm
        4. Combines the first m = min(n, k^2) matrices
        5. Normalices range to [0,255]
        6. Returns an image-like array in uint8 data-type
    """

    # ==============================
    # 1. 
    # ==============================
    vec2 = np.copy(vec)
    vec2 = np.asarray(vec, dtype=np.float64)
    n2 = vec2.size
    if n2 % 2 != 0:
        raise ValueError("Vector size must be (2n).")

    n = n2 // 2
    if k <= 0 or (k & (k - 1)) != 0:
        raise ValueError("k must be a power of 2")

    # ==============================
    # 2.
    # ==============================
    vec2[1::2] *= -1

    # ==============================
    # 3. 
    # ==============================
    v_sum = vec2[0::2] + vec2[1::2]  # tamaño n

    # ==============================
    # 4. 
    # ==============================

    def build_v_set(kpow):
        V = [np.array([1], dtype=np.int8)]
        for _ in range(1, kpow + 1):
            newV = []
            for v in V:
                newV.append(np.concatenate([v, v]))   # [v v]
                newV.append(np.concatenate([v, -v]))  # [v -v]
            V = newV
        return V

    def sign_changes_count(v):
        return int(np.count_nonzero(v[:-1] != v[1:]))

    def sequency_sort(V):
        counts = [sign_changes_count(v) for v in V]
        indices = sorted(range(len(V)), key=lambda i: (counts[i], i))
        return [V[i] for i in indices]

    def outer_product_patterns(Vordered):
        m = len(Vordered)
        return [[np.outer(Vordered[i], Vordered[j]) for j in range(m)] for i in range(m)]

    def zigzag_indices(n):
        idx = []
        for s in range(2*n - 1):
            if s % 2 == 0:
                i_start = min(s, n-1)
                i_end = max(0, s - (n-1))
                for i in range(i_start, i_end - 1, -1):
                    j = s - i
                    idx.append((i, j))
            else:
                j_start = min(s, n-1)
                j_end = max(0, s - (n-1))
                for j in range(j_start, j_end - 1, -1):
                    i = s - j
                    idx.append((i, j))
        return idx

    kpow = int(math.log2(k))
    V = build_v_set(kpow)
    Vordered = sequency_sort(V)
    grid = outer_product_patterns(Vordered)
    zz = zigzag_indices(k)
    ordered_patterns = [grid[i][j] for (i, j) in zz]

    # ==============================
    # 5. 
    # ==============================
    m = min(n, len(ordered_patterns))  
    M = np.zeros((k, k), dtype=np.float64)

    for i in range(m):
        M += v_sum[i] * ordered_patterns[i]

    # ==============================
    # 6. 
    # ==============================
    M -= M.min()
    if M.max() != 0:
        M = 255 * M / M.max()

    return M.astype(np.uint8)
def hadamard_zigzag(image_array: np.ndarray, normalize=False):
    """
    Computes the Hadamard transform for an image in jpeg algorithm

    - image_array: np.ndarray (H,W) o (H,W,3)
    - normalize: normalize coefficents?

    Returns:
        - grayscale: vector 1D following zig-zag (long n*n)
        - RGB: dict {'R':..., 'G':..., 'B':...}
    """

    def is_power_of_two(n):
        return n > 0 and (n & (n-1)) == 0

    def zigzag_indices(n):
        idx = []
        for s in range(2*n - 1):
            if s % 2 == 0:
                i_start = min(s, n - 1)
                i_end = max(0, s - (n - 1))
                for i in range(i_start, i_end - 1, -1):
                    idx.append((i, s - i))
            else:
                j_start = min(s, n - 1)
                j_end = max(0, s - (n - 1))
                for j in range(j_start, j_end - 1, -1):
                    idx.append((s - j, j))
        return idx

    def fwht_1d(a):
        a = a.astype(np.float64).copy()
        n = a.size
        h = 1
        while h < n:
            for i in range(0, n, h*2):
                for j in range(i, i+h):
                    x = a[j]
                    y = a[j+h]
                    a[j] = x + y
                    a[j+h] = x - y
            h *= 2
        return a

    def fwht_2d(mat):
        n = mat.shape[0]
        M = mat.astype(np.float64).copy()
        for i in range(n):
            M[i] = fwht_1d(M[i])
        for j in range(n):
            M[:,j] = fwht_1d(M[:,j])
        return M

    def process_channel(channel):
        n, m = channel.shape
        if n != m:
            raise ValueError("Image must be square size!")
        if not is_power_of_two(n):
            raise ValueError("Image size must be a power of 2!")
        H = fwht_2d(channel)
        if normalize:
            H = H / n
        idx = zigzag_indices(n)
        return np.array([H[i,j] for (i,j) in idx], dtype=np.float64)

    # ---------------------------
    
    # ---------------------------
    arr = np.asarray(image_array)

    # RGB
    if arr.ndim == 3 and arr.shape[2] == 3:
        return {
            'R': process_channel(arr[:,:,0]),
            'G': process_channel(arr[:,:,1]),
            'B': process_channel(arr[:,:,2]),
        }

    # grayscale
    if arr.ndim == 2:
        return process_channel(arr)

    raise ValueError("Image must be grayscale or RGB")
def plot_comparative_histogram(img1, img2, title1="Image 1", title2="Image 2"):
    """Plots a comparative intensity histogram of two images"""

    fig1, axs1 = plt.subplots(2, figsize=(10,14))

    axs1[0].hist(img1.ravel(), bins=256, range=(0, 256));
    axs1[0].set_title(title1, fontsize=title_fs)
    axs1[0].set_xlabel("Pixel value", fontsize=label_fs)
    axs1[0].set_ylabel("Frequency", fontsize=label_fs)
    axs1[1].hist(img2.ravel(), bins=256, range=(0, 256));
    axs1[1].set_title(title2, fontsize=title_fs)
    axs1[1].set_xlabel("Pixel value", fontsize=label_fs)
    axs1[1].set_ylabel("Frequency", fontsize=label_fs)

    plt.show()
def extract_windows(X, s):
    """
    Return array of shape (Us, Vs, s, s) containing all non-overlapping windows.
    """
    U, V = X.shape
    Us, Vs = U // s, V // s
    Xc = X[:Us*s, :Vs*s]

    stride0, stride1 = Xc.strides

    return as_strided(
        Xc,
        shape=(Us, Vs, s, s),
        strides=(s*stride0, s*stride1, stride0, stride1),
        writeable=False
    )
def dfa2d_vectorized(X, scales):
    """
    Fully vectorized 2D DFA.
    ----------
    X : 2D numpy array (float)
    scales : iterable of ints
    ------- Returns:
    F2s : dict {scale s : F2(s)}
    """
    X = np.asarray(X, dtype=float)
    U, V = X.shape
    F2s = {}

    for s in scales:
        Us, Vs = U // s, V // s
        if Us == 0 or Vs == 0:
            continue

        # 1. Extract all windows → (Us, Vs, s, s)
        W = extract_windows(X, s)           # shape: (Us, Vs, s, s)

        # reshape into batch dimension
        N = Us * Vs
        Wb = W.reshape(N, s, s)

        # 2. Compute cumulative sums per-window (Eq. 10)
        mean_vals = Wb.mean(axis=(1, 2), keepdims=True)
        Y = (Wb - mean_vals).cumsum(axis=1).cumsum(axis=2)   # shape (N,s,s)

        # 3. Fit plane Y_hat(i,j) = a*i + b*j + c (batched least squares)
        i, j = np.mgrid[0:s, 0:s]
        A = np.column_stack([i.ravel(), j.ravel(), np.ones(s*s)])
        # Precompute pseudoinverse once
        pinvA = np.linalg.pinv(A)           # shape (3, s*s)

        # Flatten all windows: shape (N, s*s)
        Yflat = Y.reshape(N, s*s)

        # Batch multiply: coeffs
        # result shape: (3, N)
        coeffs = pinvA @ Yflat.T
        a = coeffs[0].reshape(N, 1, 1)
        b = coeffs[1].reshape(N, 1, 1)
        c = coeffs[2].reshape(N, 1, 1)

        # 4. Reconstruct plane for all windows at once
        Yi = a * i + b * j + c   # broadcast → shape (N, s, s)

        # 5. Compute local DFA variances 
        F2_local = np.mean((Y - Yi)**2, axis=(1, 2))  # (N,)

        # 6. Average
        F2s[s] = np.sqrt(F2_local.mean())

    return F2s
def plot_dfa_with_fit(F2_dict, show=True, title="2D-DFA Fit"):
    """
    Plot the DFA scaling behavior in log-log space,
    fit a linear regression, show the fitted line and alpha.

 
    ----------
    F2_dict : dict {s : F2(s)}

    -------Returns:
    alpha : float
        Scaling exponent (slope)
    """

    # Convert to arrays
    s_vals = np.array(list(F2_dict.keys()), dtype=float)
    F_vals = np.array(list(F2_dict.values()), dtype=float)

    # Compute logs
    log_s = np.log(s_vals)
    log_F = np.log(F_vals)

    # Fit α and intercept
    alpha, intercept = np.polyfit(log_s, log_F, 1)

    # Fitted line
    log_F_fit = alpha * log_s + intercept

    if show:

      # ---- Plotting ----
      plt.figure(figsize=(6, 5))

      # Data points
      plt.scatter(log_s, log_F, label="Data", s=60)

      # Fitted line
      plt.plot(log_s, log_F_fit, "--", label=f"Fit: α = {alpha:.4f}")

      # Labels
      plt.xlabel("log(s)", fontsize=label_fs)
      plt.ylabel("log(F₂(s))", fontsize=label_fs)
      plt.title(title, fontsize=label_fs)

      plt.legend(fontsize=12)
      plt.grid(True, alpha=0.3)

      plt.show()

    return alpha
def comparative_sigma(img1, img2):
    """Recives two images and returns a pandas dataframe containing their WT MAD-based noise estimation"""
    sigma1 = estimate_sigma(img1, average_sigmas=True)
    sigma2 = estimate_sigma(img2, average_sigmas=True)

    return pd.Series({"img1":np.array([sigma1]), "img2":np.array([sigma2])})
def image_autocorrelation(img, show=True, title="Image autocorrelation"):
    """Computes the autocorrelation of a numpy 2d-array and shows a plot if wanted"""
    img = img.astype(float)

   
    F = np.fft.fft2(img)
    P = np.abs(F)**2  

    
    ac = np.fft.ifft2(P)
    ac = np.fft.fftshift(ac) 
    ac = np.real(ac)

    # normalization
    ac /= ac.max()

    if show:
        plt.imshow(ac);
        plt.colorbar()
        plt.show()
        
    return ac
def cross_correlation_2d(img1, img2, show=True, title="Cross correlation"):
    """ Computes the cross-correlation of two 2d n-arrays of same shape """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)

    
    CC = np.fft.ifft2(F1 * np.conj(F2))
    CC = np.fft.fftshift(CC)       
    CC = np.real(CC)

    
    CC /= CC.max()

    if show:
        plt.imshow(CC);
        plt.colorbar()
        plt.show()
    
    return CC
def plot_colored_scatter(x, y, c, cmap="viridis", title="Scatter Plot",
                         xlabel="x", ylabel="y", colorbar_label="color value",lims=False,identity_line=False):
    """
    Plot (x, y) as a scatter plot colored by a third variable c.

        x, y          : Lists or arrays for the coordinates.
        c             : List or array of values used to color each point.
        cmap          : Matplotlib colormap name (default: 'viridis').
        title         : Plot title.
        xlabel/ylabel : Axis labels.
        colorbar_label: Label for the colorbar.
    """
    x, y, c = np.asarray(x), np.asarray(y), np.asarray(c)

    if not (len(x) == len(y) == len(c)):
        raise ValueError(f"All lists must have the same length, got {len(x)}, {len(y)}, {len(c)}.")


    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(x, y, c=c, cmap=cmap, s=200, marker="X", alpha=0.85, edgecolors="white", linewidths=0.1,
                    vmin=c.min(), vmax=c.max())

    lo =  c.min()
    hi =  c.max()

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(colorbar_label, fontsize=11)
    cbar.set_ticks(np.linspace(lo, hi, 6)) 
    cbar.set_ticklabels([f"{v:.2f}" for v in np.linspace(lo, hi, 6)])

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(False)
    ax.set_aspect("auto")

    if lims:
        lim_min = min(x.min(), y.min())
        lim_max = max(x.max(), y.max())
        ax.set_xlim(lim_min-0.1*lim_min, lim_max+0.1*lim_max)
        ax.set_ylim(lim_min-0.1*lim_min, lim_max+0.1*lim_max)
    if identity_line:
        lim_min = min(x.min(), y.min())
        lim_max = max(x.max(), y.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="red", linestyle="--", linewidth=1.2)


    plt.tight_layout()
    plt.show()
def dip(img, device, show=True):
    """ Denoises a given image using a DIP framework
    --------------------
    img: NxN n-array
    device: str
    -------------
    NxN n-array of denoised img
    """
    net = skip(
            num_input_channels=1, num_output_channels=1,


    num_channels_down=[8, 16, 32, 64, 128],
    num_channels_up=[8, 16, 32, 64, 128],
    num_channels_skip=[0, 0, 0, 0, 0],
    upsample_mode='bilinear',
    act_fun='LeakyReLU',
    need_sigmoid=True,
    need_bias=True,
    pad='reflection'
    )

    n_img = img.astype(float)
    n_img *= 255./img.max()
    n_img -= img.min()

    y_tensor = torch.tensor(n_img).float().to(device)

    H,W = y_tensor.size()
    rnd_input = torch.rand(H,W).to(device)
    cnn = net.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if show:
        fig1, axs1 = plt.subplots(2, figsize=(20,12))

    best_loss = float('inf')
    patience = 0
    max_patience = 100
    
    train_loss = []
    steps=[]
    nrmse_list = []
    best_recon = None
    best_nrmse = float('inf')
    best_ep = None
    saves = []
    
    for step in range(1500):
        steps.append(step)
        optimizer.zero_grad()
        recon = cnn(rnd_input.unsqueeze(0).unsqueeze(0))
    
        loss = nrmse_f(recon.squeeze(), y_tensor / 255.)
    
        train_loss.append(loss.item())
    
        loss.backward()
        optimizer.step()
    
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(cnn.state_dict(), 'spi.pth')
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at step {step}")
                break
        if step % 100 == 0:
          saves.append(recon)

          if show:
          
              axs1[0].plot(steps,train_loss)
              axs1[1].imshow(torch_to_np(recon.squeeze()), cmap="gray")

    recon = cnn(rnd_input.unsqueeze(0).unsqueeze(0)).squeeze()

    return torch_to_np(recon)
def add_poisson_noise(array: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """ Takes a numpy n-array and adds poisson noise even for a distribution of negative values """
    array = np.asarray(array, dtype=float)
    if scale <= 0:
        raise ValueError("scale must be positive.")

    shift = min(0.0, array.min())   
    shifted = array - shift         

    noisy = np.random.poisson(shifted * scale).astype(float) / scale
    return (noisy + shift).reshape(array.shape)
def hadamard_matrix(n):
    """Generate an n x n orthonormal Hadamard matrix (n power of 2)."""
    if n == 1:
        return np.array([[1.0]])
    H = hadamard_matrix(n // 2)
    return (1 / np.sqrt(2)) * np.block([
        [H,  H],
        [H, -H]
    ])


def zigzag_indices(n):
    """Return zig-zag ordering indices for an n x n array."""
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(s, -1, -1):
                j = s - i
                if i < n and j < n:
                    indices.append((i, j))
        else:
            for i in range(s + 1):
                j = s - i
                if i < n and j < n:
                    indices.append((i, j))
    return indices
def hadamard_zigzag_normalized(x):
    """
    Orthonormal 2D Hadamard transform with zig-zag ordered coefficients.

    
    ----------
    x : ndarray of shape (N, N), N power of 2

    Returns
    -------
    coeffs : ndarray of shape (N*N,)
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if x.shape[0] != x.shape[1]:
        raise ValueError("Input must be square")
    if N & (N - 1) != 0:
        raise ValueError("N must be a power of 2")

    H = hadamard_matrix(N)

    
    Xh = H @ x @ H

    
    zz = zigzag_indices(N)
    return np.array([Xh[i, j] for i, j in zz])

def inverse_hadamard_zigzag(coeffs):
    """
    Inverse orthonormal 2D Hadamard transform from zig-zag coefficients.

    
    ----------
    coeffs : ndarray of shape (N*N,)
        Zig-zag ordered Hadamard coefficients

    Returns
    -------
    x : ndarray of shape (N, N)
        Reconstructed image
    """
    coeffs = np.asarray(coeffs, dtype=float)
    L = coeffs.size

    
    N = int(np.sqrt(L))
    if N * N != L:
        raise ValueError("Length of coeffs must be a perfect square")
    if N & (N - 1) != 0:
        raise ValueError("N must be a power of 2")

    
    Xh = np.zeros((N, N))
    zz = zigzag_indices(N)
    for c, (i, j) in zip(coeffs, zz):
        Xh[i, j] = c

    
    H = hadamard_matrix(N)
    x = H @ Xh @ H

    return x
def hadamard_poisson_experiment(img, lvl=1.,clip=True, show=True):
    """
    Simulate Poisson noise in Hadamard acquisition and reconstruct image.

    ----------
    img : ndarray (N, N), N power of 2
        Input image
    peak : float
        Expected photon count (controls noise level)
    clip : bool
        Clip reconstructed image to original min/max
    show: bool
        if true will show a graphic that resumes all results


    ------- Returns:
    image_rec : ndarray (N, N)
        Reconstructed noisy image
    coeffs_clean : ndarray (N*N,)
        Clean Hadamard coefficients
    coeffs_noisy : ndarray (N*N,)
        Noisy Hadamard coefficients
    """
    def plot_hadamard_experiment_results(img, img_noisy, c_clean, c_noisy):
        
        title_fs = 24
        label_fs = 18
        tick_fs = 18
    
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")
        plt.title("Original image", fontsize=title_fs)
        plt.axis("off")
        plt.show()
    
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_noisy, cmap="gray")
        plt.title("Reconstruction (Poisson noise)", fontsize=title_fs)
        plt.axis("off")
        plt.show()
    
        
        plt.figure(figsize=(5, 5))
        im = plt.imshow(img_noisy - img, cmap="gray")
        plt.title("Difference", fontsize=title_fs)
        plt.axis("off")
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=tick_fs)
        plt.show()
    
        
        plt.figure(figsize=(7, 4))
        plt.plot(c_clean)
        plt.title("Clean Hadamard coeffs", fontsize=title_fs)
        plt.xlabel("Index", fontsize=label_fs)
        plt.ylabel("Value", fontsize=label_fs)
        plt.tick_params(axis="both", labelsize=tick_fs)
        plt.tight_layout()
        plt.show()
    
        
        plt.figure(figsize=(7, 4))
        plt.plot(c_noisy)
        plt.title("Noisy Hadamard coeffs", fontsize=title_fs)
        plt.xlabel("Index", fontsize=label_fs)
        plt.ylabel("Value", fontsize=label_fs)
        plt.tick_params(axis="both", labelsize=tick_fs)
        plt.tight_layout()
        plt.show()
    
        
        plt.figure(figsize=(7, 4))
        plt.plot(c_noisy - c_clean)
        plt.title("Injected noise (Δ coeffs)", fontsize=title_fs)
        plt.xlabel("Index", fontsize=label_fs)
        plt.ylabel("Value", fontsize=label_fs)
        plt.tick_params(axis="both", labelsize=tick_fs)
        plt.tight_layout()
        plt.show()
    
    image = np.asarray(img, dtype=float)

    # 1. Forward Hadamard transform
    coeffs_clean = hadamard_zigzag_normalized(img)

    # 2. Poisson noise in acquisition domain
    coeffs_noisy = add_poisson_noise(coeffs_clean, scale=lvl)

    # 3. Inverse Hadamard transform
    image_rec = inverse_hadamard_zigzag(coeffs_noisy)

    if clip:
        image_rec = np.clip(
            image_rec,
            img.min(),
            img.max()
        )
    if show:
        plot_hadamard_experiment_results(img, image_rec, coeffs_clean, coeffs_noisy)

    return pd.Series({"corrupted":image_rec, "coeff":coeffs_clean, "corrupted_coeff":coeffs_noisy})
    
def numerical_evaluation(img, noise_lvls, device, referenced=True, no_referenced=True, show=True):
    """ Runs hadamard poisson experiment and reconstruction for different lvls of noise
     ----------
    img : ndarray (N, N), N power of 2
        Input image
    noise_lvls : array
        array of values for poisson corruption
    referenced : bool
       If true will evaluate reference metrics
    no_referenced : bool
        If true will evaluate no-reference metrics
    show: bool
        if true will show a graphics of every_subprocess
    """
    
    results = list()
    for noise_lvl in noise_lvls:
        track = pd.Series(dtype=object)
        track["noise_lvl"] = noise_lvl
        
        poisson_run = hadamard_poisson_experiment(img=img,lvl=noise_lvl,clip=True,show=show)
        track["corrupted"] = poisson_run["corrupted"].copy()

        denoised = dip(img=poisson_run["corrupted"], device=device, show=show)
        track["denoised"] = denoised.copy()

        if referenced:
            track["ssim_corrupted"] = ssim(img, track["corrupted"], data_range=img.max() - img.min())
            track["ssim_denoised"] = ssim(img, track["denoised"], data_range=img.max() - img.min())

            track["psnr_corrupted"] = psnr(img, track["corrupted"], data_range=img.max() - img.min())
            track["psnr_denoised"] = psnr(img, track["denoised"], data_range=img.max() - img.min())
        if no_referenced:
            residue = poisson_run["corrupted"].copy() - denoised.copy()
            track["residue"] = residue.copy()

            ac = image_autocorrelation(track["residue"], show=show)
            cc = cross_correlation_2d(track["denoised"], track["residue"], show=show)

            track["ac_var"] = np.var(ac)
            track["cc_var"] = np.var(cc)

            track["sigma_corrupted"] = estimate_sigma(track["corrupted"], average_sigmas=True)
            track["sigma_denoised"] = estimate_sigma(track["denoised"], average_sigmas=True)

            scales = np.arange(6,16,1)
            track["alpha_corrupted"] = plot_dfa_with_fit(dfa2d_vectorized(track["corrupted"], scales),show=show)
            track["alpha_denoised"] = plot_dfa_with_fit(dfa2d_vectorized(track["denoised"], scales),show=show)

        results.append(track)

    return pd.DataFrame(results)