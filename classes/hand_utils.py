
def no_landmarks_wrapper(fun):
    """A function wrapped with this will skip itself and
    return None if `self.landmarks` is None"""

    def wrapped_fun(*args, **kwargs):
        if args[0].landmarks is None:
            return None
        else:
            return fun(*args, **kwargs)

    return wrapped_fun

def get_distance(point1, point2):
    # point = tuple(int, int)
    translated_point = (point1[0] - point2[0], point1[1] - point2[1])
    return np.sqrt(translated_point[0] ** 2 + translated_point[1] ** 2)


def get_consecutive_ldk_distances(_dict_landmarks, _landmark_ids_list):
    total_distance = 0
    for idx, ldk_id in enumerate(_landmark_ids_list[:-1]):
        next_ldk_id = _landmark_ids_list[idx + 1]
        distance = get_distance(_dict_landmarks[ldk_id], _dict_landmarks[next_ldk_id])
        # print(distance, ldk_id, next_ldk_id)
        total_distance += distance
    return total_distance
