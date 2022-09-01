from math import inf, sqrt

def get_statistics(labels, num_classes):
    """How many samples per class we got from data preprocessing
    """
    stats = []
    percentages = []
    total = len(labels)
    for i in range(num_classes):
        num_samples = labels.count(i)
        stats.append(num_samples)
        percentages.append(round(num_samples/total, 2))
    return stats, percentages

def distance(rectA, rectB):
    """Compute distance from two given bounding boxes
    """
    
    # check relative position
    left = (rectB[2] - rectA[0]) <= 0
    bottom = (rectA[3] - rectB[1]) <= 0
    right = (rectA[2] - rectB[0]) <= 0
    top = (rectB[3] - rectA[1]) <= 0
    
    vp_intersect = (rectA[0] <= rectB[2] and rectB[0] <= rectA[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rectA[1] <= rectB[3] and rectB[1] <= rectA[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 
    
    if rect_intersect:
        return 0
    elif top and left:
        return int(sqrt((rectB[2] - rectA[0])**2 + (rectB[3] - rectA[1])**2))
    elif left and bottom:
        return int(sqrt((rectB[2] - rectA[0])**2 + (rectB[1] - rectA[3])**2))
    elif bottom and right:
        return int(sqrt((rectB[0] - rectA[2])**2 + (rectB[1] - rectA[3])**2))
    elif right and top:
        return int(sqrt((rectB[0] - rectA[2])**2 + (rectB[3] - rectA[1])**2))
    elif left:
        return (rectA[0] - rectB[2])
    elif right:
        return (rectB[0] - rectA[2])
    elif bottom:
        return (rectB[1] - rectA[3])
    elif top:
        return (rectA[1] - rectB[3])
    else: return inf

def normalize(features, size, maxw, maxh):
    """Normalize bounding boxes given the pdf size
    """
    max_area = maxw*maxh
    for feat in features:
        feat[0] = feat[0]/maxw
        feat[1] = feat[1]/maxh
        feat[2] = feat[2]/size[0]
        feat[3] = feat[3]/size[1]
        feat[4] = feat[4]/max_area
        feat[5] = feat[5]/size[0]
        feat[6] = feat[6]/size[1]
        feat[7] = feat[7]/size[0]
        feat[8] = feat[8]/size[1]
        
    return features

def center(rect):
    return [int(rect[2]-(rect[2]-rect[0])/2), int(rect[3]-(rect[3]-rect[1])/2)]
