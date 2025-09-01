import numpy as np
import torch
import h5py

def read_h5_events(hdf_path):
    """
    Read events from HDF5 file into 4xN numpy array (N=number of events)
    """
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        events = np.stack(( f['events/x'][:], 
                            f['events/y'][:], 
                            f['events/ts'][:], 
                            np.where(f['events/p'][:], 1, -1)), axis=1)
    else:
        events = np.stack(( f['events/xs'][:], 
                            f['events/ys'][:], 
                            f['events/ts'][:], 
                            np.where(f['events/ps'][:], 1, -1)), axis=1)
    return events

def read_h5_image(hdf_path):
    """
    Read image from HDF5 file
    """

    images = []

    with h5py.File(hdf_path, 'r') as f:
        sharp_images = f['sharp_images']
        for key in sharp_images.keys():
            img = sharp_images[key][:]
            images.append(img)

    return images


def extract_events(event, dataset):
    if dataset == 'hs_ergb':
        x = event['x']
        y = event['y']
        t = event['t']
        p = event['p']
    else:
        x = event['x']
        y = event['y']
        t = event['timestamp']
        p = event['polarity']
    return np.stack([t, x, y, p], axis=1)  # shape: (N, 4)


## bs_ergb x, y overflow fix
# https://github.com/ercanburak/EVREAL/blob/398f03551b7f5d150ae10aa89e4e731557fce240/tools/bs_ergb_to_npy.py#L12
def convert_and_fix_event_pixels(data, upper_limit, fix_overflows=True):
    data = data.astype(np.int32)
    overflow_indices = np.where(data > upper_limit*32)
    num_overflows = overflow_indices[0].shape[0]
    if fix_overflows and num_overflows > 0:
        data[overflow_indices] = data[overflow_indices] - 65536
    data = data / 32.0
    data = np.rint(data)
    data = data.astype(np.int16)
    data = np.clip(data, 0, upper_limit)
    return data

## bs_ergb voxel grid
def events_to_voxel_grid(events, num_bins, width, height):
    assert events.shape[1] == 4
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    
    last_stamp, first_stamp = events[-1, 0], events[0, 0]
    deltaT = max(last_stamp - first_stamp, 1.0)

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts, xs, ys, pols = events[:, 0], events[:, 1].astype(int), events[:, 2].astype(int), events[:, 3]
    pols[pols == 0] = -1

    # 인덱스 범위 제한
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    tis = np.clip(ts.astype(int), 0, num_bins - 1)

    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = (tis >= 0) & (tis < num_bins)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis >= 0) & ((tis + 1) < num_bins)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    return voxel_grid.reshape((num_bins, height, width))

## vimeo90k voxel grid
def events_to_voxel_grid_vimeo90k(events, num_bins, width, height):
    assert events.shape[1] == 4
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    
    last_stamp, first_stamp = events[-1, 0], events[0, 0]
    deltaT = last_stamp - first_stamp

    print(f"first_stamp : {first_stamp}")
    print(f"last_stamp : {last_stamp}")
    print(f"deltaT : {deltaT}")

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts, xs, ys, pols = events[:, 0], events[:, 1].astype(int), events[:, 2].astype(int), events[:, 3]
    pols[pols == 0] = -1

    # 인덱스 범위 제한
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    tis = np.clip(ts.astype(int), 0, num_bins - 1)

    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = (tis >= 0) & (tis < num_bins)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis >= 0) & ((tis + 1) < num_bins)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    return voxel_grid.reshape((num_bins, height, width))




## hs_ergb voxel grid
# https://github.com/iCVTEAM/LETGAN/blob/d632b3845357559361f628d1b250bce60f95fe07/utils/event_utils.py#L187
def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search sorted pytorch tensor
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    W, H = img.shape # 1440, 1080

    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)

    return img


def events_to_image_torch(x, y, p, device, width, height, clip_out_of_range=False, interpolation='bilinear', padding=False):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param padding if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """

    if interpolation == 'bilinear' and padding:
        img_size = (height+1,width+1)
    else:
        img_size = list((height,width))

    mask = torch.ones(x.shape, device=device)

    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(x>=clipx, zero_v, ones_v) * torch.where(y>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size).to(device)

    if interpolation == 'bilinear' and x.dtype is not torch.long and x.dtype is not torch.long:
        pxs = (x.floor()).float()
        pys = (y.floor()).float()
        dxs = (x-pxs).float()
        dys = (y-pys).float()
     
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = p.squeeze() * mask

        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)

    else:
        if x.dtype is not torch.long:
            x = x.long().to(device)
        if y.dtype is not torch.long:
            y = y.long().to(device)

        img.index_put_((y, x), p, accumulate=True)

    return img


def events_to_voxel_torch(events_data, num_bins, width, height, temporal_bilinear=True, device=None):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    t = events_data[:, 0]
    x = events_data[:, 1]
    y = events_data[:, 2]
    p = events_data[:, 3]

    assert(len(x)==len(y) and len(y)==len(t) and len(t)==len(p))

    bins = []
    dt = t[-1]-t[0]
    
    # dt가 0인 경우 처리 (모든 이벤트가 동일한 타임스탬프)
    if dt == 0:
        # 모든 bin에 동일하게 분배
        for bin in range(num_bins):
            vb = events_to_image_torch(x, y, p, device=None, width=width, height=height, clip_out_of_range=True, padding=False)
            bins.append(vb)
        bins = torch.stack(bins)
        return bins
    
    t_norm = (t-t[0]) / dt * (num_bins-1) # timestamp normalization : [0, B-1]로 정규화

    t_norm = torch.tensor(t_norm)
    p = torch.tensor(p) 
    x = torch.tensor(x)
    y = torch.tensor(y)

    zeros = torch.zeros(t_norm.shape)
    
    for bin in range(num_bins):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bin)) # 양방향 linear interpolation 가중치

            weights = p * bilinear_weights
            vb = events_to_image_torch(x, y, weights, device=None, width=width, height=height, clip_out_of_range=True, padding=False)
        
        else:
            tstart = t[0] + dt + bin
            tend = tstart + dt

            beg = binary_search_torch_tensor(t, 0, len(t)-1, tstart) 
            end = binary_search_torch_tensor(t, 0, len(t)-1, tend) 

            vb = events_to_image_torch(x[beg:end], y[beg:end], p[beg:end], device=None, width=width, height=height, clip_out_of_range=True, padding=False)
   
        bins.append(vb)
    
    bins = torch.stack(bins)
    return bins

'''
t_norm=1.8인 이벤트가 있다면:
    bi=1: 가중치 = max(0, 1.0-|1.8-1|) = max(0, 0.2) = 0.2
    bi=2: 가중치 = max(0, 1.0-|1.8-2|) = max(0, 0.8) = 0.8
    bi=0,3,4...: 가중치 = 0 (거리가 멀어서)

이벤트가 특정 bin과 거리가 멀면 가중치가 음수가 되는데, 이를 0으로 만들어서 해당 bin에 기여하지 않도록 함
결과적으로 각 이벤트는 가까운 1-2개 bin에만 가중치를 분배하게 됨
'''

if __name__ == "__main__":
    
    # 27개의 랜덤 x, y 좌표 (0~100)
    x = np.random.randint(0, 101, 27)
    y = np.random.randint(0, 101, 27)
    
    # 1부터 28까지 27개의 타임스탬프
    t = np.arange(1, 28)
    
    # 27개의 랜덤 극성값 (0 또는 1)
    p = np.random.randint(0, 2, 27)

    print("x:", x)
    print("y:", y) 
    print("t:", t)
    print("p:", p)

    events_data = np.zeros((x.shape[0], 4), dtype=np.float32)  # [t, x, y, p]
    events_data[:, 0] = t  # timestamp
    events_data[:, 1] = x  # x coordinate
    events_data[:, 2] = y  # y coordinate  
    events_data[:, 3] = p  # polarity
    
    bins = events_to_voxel_torch(events_data, num_bins=10, width=100, height=100)