def sanity(laneTrack):
    min_dist_tresh = 2.75
    max_dist_tresh = 3.25

    if  laneTrack.min_distance < min_dist_tresh or laneTrack.min_distance > max_dist_tresh:
        laneTrack.detected = False

    return laneTrack.detected, laneTrack
