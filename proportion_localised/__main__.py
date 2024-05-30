# © 2024 Alexander Taylor
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import numpy as np
from skimage import measure 
import scipy.spatial.distance as dist
from skimage import measure
from typing import Union
from torch import Tensor
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import cv2

def get_rotated_bounding_box_parameters(trial_mask):
    '''
    When given a mask, this function will return the parameters of the minimum area
    rotated bounding box which can contain the mask.
    '''
    contours, _ = cv2.findContours(trial_mask.astype(np.uint8)*255, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    try:                
        all_points = np.concatenate([contour[:,0,:] for contour in contours])
        hull = cv2.convexHull(all_points)
    except:
        hull = contours[np.argmax([len(item) for item in contours])]
        
    ((center_x, center_y), (width, height), angle_of_rotation) = cv2.minAreaRect(hull)
    return ((center_x, center_y), (width, height), angle_of_rotation)

def create_bounding_box_mask(mask_shape, bounding_box_info):
    '''
    Given a mask shape and the bounding box parameters, 
    this function will return a mask with the bounding box filled
    '''
    box = cv2.boxPoints(bounding_box_info)
    box = np.int0(box)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 255, -1)
    return mask

def check_overlap_rotated(box1, box2):
    '''
    Given two rotated bounding boxes, this function will return the 
    percentage of of the second box with respect to the first box.
    (The larger box is always given as the first)
    '''
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    # Calculate intersection area
    intersection_area = poly1.intersection(poly2).area
    
    # Calculate area of the first bounding box
    box1_area = poly1.area
    
    # Check if intersection area is at least 50% of the first bounding box area
    return (intersection_area/box1_area)

## get the min distance between the centers - cdist
def get_min_distance_indices(coordinates):
    '''
    Given a list of coordinates, (the centers of each anomaly),
    this function will return the a list of the paris indices which correspond to the 
    centers which are closest to each other
    '''
    distances = cdist(coordinates, coordinates)

    np.fill_diagonal(distances, np.inf)

    # Find the indices of the closest coordinates
    closest_indices = np.argmin(distances, axis=1)
    min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    min_index
    
    out = []
    for a, b in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
        if a!=b:
            out.append((a, b))

    return np.array(out[::2])

def build_graph(pairs):
    graph = {}
    for pair in pairs:
        a, b = pair
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
    return graph

def dfs(graph, start, visited, component):
    visited.add(start)
    component.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, component)

def connected_components(pairs):
    graph = build_graph(pairs)
    visited = set()
    components = []
    for vertex in graph:
        if vertex not in visited:
            component = []
            dfs(graph, vertex, visited, component)
            components.append(component)
    return components

def sort_linked_sets(pairs):
    '''
    This takes a list of pairs, where each pair is a list of two indices,
    and those two indices represent the indices of the bounding boxes which are overlapping to each other.
    It returns a list of lists, where each internal list contains the indices of the bounding boxes which are overlapping with each other
    '''
    components = connected_components(pairs)
    return [sorted(component) for component in components]

def make_scaled_bb(temp_mask, min_target_scale):
    '''
    When given a mask, and the minimum target scale, this function will extract the rotated bounding box parameters,
    and if needed, scale the mask such that it meets the minimum target scale requirements
    '''
    ((center_x, center_y), (width, height), angle_of_rotation) = get_rotated_bounding_box_parameters(temp_mask)
    
    if width<256//min_target_scale:
        width = 256//min_target_scale

    if height<256//min_target_scale:
        height = 256//min_target_scale
        
    return ((center_x, center_y), (width, height), angle_of_rotation)

def get_overlapping_masks(bbs, overlap_limit=0.5):
    '''
    Given a list of rotated bounding box parameters, this returns a list of lists, 
    where each internal list contains the indices of the bounding boces which contain>overlap_limit
    with each other

    '''
    shape = (256, 256)
    bounding_box_infos = [item for item in bbs]
    
    coordinates = [item[0] for item in bounding_box_infos]
    dims = [item[1] for item in bounding_box_infos]
    out = get_min_distance_indices(coordinates)
    merge_list = []
    for indices in out:
        # box1, box2 = coordinates[indices[0]], coordinates[indices[1]]
        box1area = sum(dims[indices[0]])
        box2area = sum(dims[indices[1]])

        bounding_box_info1 = cv2.boxPoints(bounding_box_infos[indices[0]])
        bounding_box_info2 = cv2.boxPoints(bounding_box_infos[indices[1]])

        if box1area<box2area:
            # out = np.zeros(shape)
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2
            # out[out>0] = 255

            overlap = check_overlap_rotated(bounding_box_info1, bounding_box_info2)

        else:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            overlap = check_overlap_rotated(bounding_box_info2, bounding_box_info1)
        
        if overlap>overlap_limit:
            # out = np.zeros(shape)
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            # new_box = get_rotated_bounding_box_parameters(out)

            # new_bounding_box_infos = [item for i, item in enumerate(bounding_box_infos) if i not in [indices[0], indices[1]]]
            merge_list.append([indices[0], indices[1]])

            #restart = True
            #break
      
    groups = sort_linked_sets(merge_list)
    
    for indice in range(len(coordinates)):
        if not np.any([indice in group for group in groups]):
            groups.append([indice])
    return groups

def merge_overlapping(individual_masks, scaled_bbs, min_target_scale, overlap_limit=0.5):
    '''
    Given a list of individual masks and the corresponding bounding box parameters, the minimum target scale and the overlap limit,
    this will return a list of bounding box parameters which are the result of merging the masks which overlap with each other.
    We find two iterations of overlapping work best
    '''
    groups = get_overlapping_masks(scaled_bbs, overlap_limit=overlap_limit)
    out_masks = []
    for group in groups:
        out_mask = np.zeros((256, 256))
        for ind in group:
            out_mask += individual_masks[ind]
        out_masks.append(out_mask)
        
    scaled_bbs_2 = [make_scaled_bb(item, min_target_scale) for item in out_masks]
    
    groups = get_overlapping_masks(scaled_bbs_2, overlap_limit=overlap_limit)
    out_masks_2 = []
    for group in groups:
        out_mask = np.zeros((256, 256))
        for ind in group:
            out_mask += out_masks[ind]
        out_masks_2.append(out_mask)
    scaled_bbs_3 = [make_scaled_bb(item, min_target_scale) for item in out_masks_2]
    
    return scaled_bbs_3

def target_to_rotated_scaled_merged_BBs(target, overlap_limit, min_target_scale, shape=(256, 256)):
    '''
    Given a target mask (which may contain many non-overlapping anomalies), this function will return a list of bounding box parameters
    of each individual anomaly, given the overlap_limit, and the minimum target scale 
    '''
    regions = [region for region in measure.regionprops(measure.label(target>0))]
    
    if len(regions)==1:
        return [make_scaled_bb(target, min_target_scale)]
    scaled_bbs = []
    individual_masks = []
    for region_ind, region in enumerate(regions):
        temp_mask = np.zeros((256, 256))
        temp_mask[tuple([[item[0] for item in region.coords], [item[1] for item in region.coords]])] = 1
        individual_masks.append(temp_mask)
        scaled_bbs.append(make_scaled_bb(temp_mask, min_target_scale))
    
    return merge_overlapping(individual_masks, scaled_bbs, min_target_scale, overlap_limit=overlap_limit)

def PL_calculation(heatmaps, targets, min_target_scale=8, n_thresholds=25, 
                       overlap_limit=1/3,
                       ):
    IoUs = []
    
    min_pred = heatmaps.min()
    max_pred = heatmaps.max()
    
    if not isinstance(heatmaps, np.ndarray):
        heatmaps = heatmaps.numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    
    thresholds = [threshold for threshold in np.linspace(min_pred, max_pred, n_thresholds+2)[1:-1]]
    
    all_prediction_masks = []
    all_temp_masks = []
    all_bounding_box_infos = []

    IoUs = []
    for heatmap, target in zip(heatmaps, targets):
        bounding_box_infos = target_to_rotated_scaled_merged_BBs(target, overlap_limit, min_target_scale, shape=(256, 256))
        xx, yy = np.meshgrid(np.arange(256), np.arange(256))
        
        prediction_masks = []
        temp_masks = []
            
        all_bounding_box_infos.append(bounding_box_infos)
        
        region_centers_ = []
        for bounding_box_info in bounding_box_infos:
            center_x, center_y = bounding_box_info[0]
            region_centers_.append((center_y, center_x))
            
        distances = dist.cdist(np.array([xx.ravel(), 
                                         yy.ravel()]).T, 
                               region_centers_, 
                               metric='euclidean').reshape(256, 256, len(region_centers_))
                
        for region_ind, bounding_box_info in enumerate(bounding_box_infos):
            
            bb_mask = create_bounding_box_mask((256, 256), 
                                               bounding_box_info)
            
            closest_pixels = np.argmin(distances.swapaxes(0, 1), axis=-1)==region_ind
            prediction_mask = np.zeros((256, 256))
            prediction_mask[closest_pixels] = 1
            prediction_mask[bb_mask>0] = 1

            prediction_masks.append(prediction_mask)
            temp_masks.append(bb_mask)
        
        all_prediction_masks.append(prediction_masks)
        all_temp_masks.append(temp_masks)

        # loop over the prediction_masks
        for prediction_mask, temp_mask in zip(prediction_masks, temp_masks):
            IoUs_per_heatmap_overthresholds = []
            for threshold in thresholds:
                nov_threshold = heatmap>threshold
                predictions = np.logical_and(nov_threshold, prediction_mask)
                IoU = calculate_IoU(predictions=predictions, mask=temp_mask)
                IoUs_per_heatmap_overthresholds.append(IoU)
            IoUs.append(IoUs_per_heatmap_overthresholds)

    array_out = np.array(IoUs).T
    array_out_ = (array_out>0.3).mean(axis=1)
    thres_ind = array_out_.argmax()
    thres_value = thresholds[thres_ind]

    return array_out, thres_value, thres_ind, ((all_prediction_masks, all_temp_masks, all_bounding_box_infos))
                        
def calculate_IoU(predictions, mask):
    '''
    Given a mask and a prediction, returns the IoU
    '''
    return (np.logical_and(predictions, mask)).sum()/(np.logical_or(predictions, mask)).sum()

class PL:
    '''
    PL class, this calculates the PL score, and also stores the data used in the calculation
    '''
    def __init__(self, 
                 anomaly_likelihood_definitely_increasing: bool = False,):
        self.anomaly_likelihood_definitely_increasing = anomaly_likelihood_definitely_increasing
        self.dictionary = {}
    
    def calculate(self, 
                  preds: Union[np.ndarray, Tensor], 
                  ground_truths: Union[np.ndarray, Tensor], 
                  iou_limit: float = 0.3,
                  min_target_scale_inverse: int = 8, 
                  overlap_limit: float = 1/3,
                  n_thresholds: int = 25) -> Union[float, tuple]:
        
        dict_key = f"{min_target_scale_inverse}_{n_thresholds}_{overlap_limit}"
        
        if dict_key not in self.dictionary:
            data_out_anomalies_increasing = PL_calculation(preds, 
                                                            ground_truths,
                                                            min_target_scale=min_target_scale_inverse, 
                                                            n_thresholds=n_thresholds,
                                                            overlap_limit=overlap_limit,
                                                            )
            score_anomales_increasing = (data_out_anomalies_increasing[0]>iou_limit).mean(axis=1).max()

            if self.anomaly_likelihood_definitely_increasing:
                score = score_anomales_increasing  
                array = data_out_anomalies_increasing[0]
                data_out_pl = data_out_anomalies_increasing
            else:
                data_out_anomalies_decreasing = PL_calculation(-1*preds, 
                                                ground_truths,
                                                min_target_scale=min_target_scale_inverse, 
                                                n_thresholds=n_thresholds,
                                                overlap_limit=overlap_limit,)
                score_anomales_decreasing = (data_out_anomalies_decreasing[0]>iou_limit).mean(axis=1).max()

                if score_anomales_increasing>score_anomales_decreasing:
                    score = score_anomales_increasing
                    array = data_out_anomalies_increasing[0]
                    data_out_pl = data_out_anomalies_increasing
                else:
                    score = score_anomales_decreasing
                    array = data_out_anomalies_decreasing[0]
                    data_out_pl = data_out_anomalies_decreasing

            self.dictionary[dict_key] = array
            self.dictionary[dict_key+"_data"] = (data_out_pl[1], data_out_pl[2], data_out_pl[3]) 
        else:
            array = self.dictionary[dict_key]
            score = (array>iou_limit).mean(axis=1).max()
            
            data_out_threshold_masks = self.dictionary[dict_key+"_data"] 
            data_out_pl = (array, data_out_threshold_masks[0], data_out_threshold_masks[1], data_out_threshold_masks[2])
        return score, data_out_pl
    
    def reset(self):
        self.dictionary = {}

def get_PL_instance(anomaly_likelihood_definitely_increasing: bool = False) -> PL:
    '''
    Returns a instance of the PL class, this gives the user more flexibility as they have access to
    the underlying class and its data
    '''
    return PL(anomaly_likelihood_definitely_increasing=anomaly_likelihood_definitely_increasing)

def calculate(preds: Union[np.ndarray, Tensor], 
              ground_truths: Union[np.ndarray, Tensor], 
              iou_limit: float = 0.3,
              min_target_scale: float = 1/8,
              overlap_limit: float = 1/3,
              n_thresholds: int = 25,
              anomaly_likelihood_definitely_increasing: bool = False,
              return_score_only: bool = False) -> Union[float, tuple]:
    '''
    Calculates and returns the PL score
    Only set anomaly_likelihood_definitely_increasing to True
    if you know that lower predictions represent no anomaly
    and higher represents anomaly. Setting to True will double the speed, as we 
    don't have to check both directions.
    If you are unsure which way round the predictions are, set it to False (default),
    so both directions are checked
    '''
    score, data =  PL(anomaly_likelihood_definitely_increasing=anomaly_likelihood_definitely_increasing
                      ).calculate(preds, 
                                  ground_truths, 
                                  iou_limit,
                                  min_target_scale_inverse=min_target_scale**-1,
                                  overlap_limit=overlap_limit,
                                  n_thresholds=n_thresholds)
    if return_score_only:
        return score
    else:
        return score, data
    
def tensor_to_cv2(tensor):
    image = tensor.cpu().numpy()

    if image.shape[0] == 1 or image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    image = (image * 255).astype('uint8')

    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = np.dstack([image]*3)
        np.moveaxis(image, 0, -1)
    return image[:, :, ::-1]  

def get_rotation_info(trial_mask):
    contours, _ = cv2.findContours(trial_mask.astype(np.uint8)*255, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    try:                
        all_points = np.concatenate([contour[:,0,:] for contour in contours])
        hull = cv2.convexHull(all_points)
    except:
        import traceback
        print(traceback.format_exc())
        print("contours")
        print(contours)
        print(len(contours))
        print("all_points")
        print(len(all_points))
        print(all_points.dtype)
        print(len(contours))
        hull = contours[np.argmax([len(item) for item in contours])]
        
    ((center_x, center_y), (width, height), angle_of_rotation) = cv2.minAreaRect(hull)
    return ((center_x, center_y), (width, height), angle_of_rotation)

def drawcontour(image, bounding_box_info, thickness=5, color=(0, 0, 255)):
    box = cv2.boxPoints(bounding_box_info)
    box = np.intp(box)
    cv2.drawContours(image, [box], -2, color, thickness)
    
def calculate_IoU(predictions, mask):
    return (np.logical_and(predictions, mask)).sum()/(np.logical_or(predictions, mask)).sum()

def make_demo_image(input_image, h_map_w_threshold, prediction_mask, bb_info):
    '''
    Takes the input image, the heatmap with threshold applied, the prediction mask,
    and the bounding box info of the target
    returns an image with the mask applied to the input image, the thresheld heatmap
    drawn on the image as a red tint, and the rotated bounding box drawn in blue
    '''
    im = tensor_to_cv2(input_image)
    im = im*prediction_mask[:,:,None].astype(np.uint8)
    gray_out = im==0
    
    mask_bgr = cv2.merge([h_map_w_threshold.astype(np.uint8), 
                          h_map_w_threshold.astype(np.uint8), 
                          h_map_w_threshold.astype(np.uint8)])
    
    alpha = 0.9  # Adjust as needed

    red_shade = np.zeros_like(mask_bgr, dtype=np.uint8)
    red_shade[:, :, 0] = 255  # Set the red channel to 255
    red_shade[mask_bgr==0] = 0
    
    im[:,:,0][mask_bgr[:,:,0]>0] = im[:,:,0][mask_bgr[:,:,0]>0]*(1-alpha)

    im = cv2.addWeighted(im, 1, red_shade, alpha, 0,)
    
    im[gray_out] = 200
    
    drawcontour(im, bb_info)
    
    return im

def visualise_PL(input_images,
                  predictions,
                  ground_truths, 
                  iou_array,
                  thres_value, 
                  thres_ind,
                  all_prediction_masks, 
                  all_anomaly_masks, 
                  all_bounding_box_infos,
                  specific_indices=[]):
    '''
    Visualise the input images, predictions, ground truths, input images with mask, prediction, and bounding box applied, 
    and the ground truth mask
    '''
    import matplotlib.pyplot as plt
    
    if len(specific_indices)>0:
        plot_indices = specific_indices
    elif len(specific_indices)==0 and isinstance(specific_indices, list):
        print("Note: length of specific_indices is 0 and specific_indices is not a list, plotting none")
        plot_indices = specific_indices
    else:
        plot_indices = np.arange(iou_array.shape[0])
        
    ious = iou_array[thres_ind]

    plot_ims_false = []
    ii = -1
    for image_ind, (h_map, input_image, target, prediction_masks, anomaly_masks, bb_infos) in enumerate(zip(predictions, 
                                                                                             input_images,
                                                                                             ground_truths, 
                                                                                             all_prediction_masks, 
                                                                                             all_anomaly_masks, 
                                                                                             all_bounding_box_infos)):


        for prediction_mask, anomaly_mask, bb_info in zip(prediction_masks, anomaly_masks, bb_infos):
            ii+=1
            
            if not ii in plot_indices:
                continue

            iou = ious[ii]

            # IoU = calculate_IoU(predictions=np.logical_and(nov_threshold, prediction_mask), mask=anomaly_mask)

            im = make_demo_image(input_image,
                                 (h_map>thres_value).numpy(), 
                                 prediction_mask, 
                                 bb_info)
            
            fig, axes = plt.subplots(1, 5, figsize=(15, 75))

            axes[0].imshow(tensor_to_cv2(input_image))
            axes[1].imshow(h_map)
            axes[2].imshow(target)
            axes[3].imshow(im)
            axes[4].imshow(anomaly_mask)

            axes[0].set_title(f"input image: {image_ind}, anomaly: {ii}")
            axes[1].set_title("prediction")
            axes[2].set_title("ground_truth")
            axes[3].set_title(f"masked anomaly, IoU: {iou:.3g}")
            axes[4].set_title(f"masked anomaly BB")
            
            plt.axis('off')
                
            plot_ims_false.append(im)