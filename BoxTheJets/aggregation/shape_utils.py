import numpy as np


def get_box_edges(x, y, w, h, a):
    '''
        Return the corners of the box given one corner, width, height
        and angle

        Inputs
        ------
        x : float
            Box left bottom edge x-coordinate
        y : float
            Box left bottom edge y-coordinate
        w : float
            Box width
        h : float
            Box height
        a : flat
            Rotation angle

        Outputs
        --------
        corners : numpy.ndarray
            Length 4 array with coordinates of the box edges
    '''
    cx = (2 * x + w) / 2
    cy = (2 * y + h) / 2
    centre = np.array([cx, cy])
    original_points = np.array(
        [
            [cx - 0.5 * w, cy - 0.5 * h],  # This would be the box if theta = 0
            [cx + 0.5 * w, cy - 0.5 * h],
            [cx + 0.5 * w, cy + 0.5 * h],
            [cx - 0.5 * w, cy + 0.5 * h],
            # repeat the first point to close the loop
            [cx - 0.5 * w, cy - 0.5 * h]
        ]
    )
    rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    corners = np.matmul(original_points - centre, rotation) + centre
    return corners


def get_point_distance(x0, y0, x1, y1):
    '''
        Get Euclidiean distance between two points (x0, y0) and (x1, y1)

        Inputs
        ------
        x0 : float
            First point x-coordinate
        y0 : float
            First point y-coordinate
        x1 : float
            Second point x-coordinate
        y1 : float
            Second point y-coordinate

        Outputs
        --------
        dist : float
            Euclidian distance between (x0, y0) and (x1, y1)
    '''
    return np.sqrt((x0 - x1)**2. + (y0 - y1)**2.)


def get_box_distance(box1, box2):
    '''
        Get point-wise distance betweeen 2 boxes.
        Calculates and find the average distance between each edge
        for each box

        Inputs
        ------
        box1 : list
            parameters corresponding to the first box (see `get_box_edges`)
        box2 : list
            parameters corresponding to the second box (see `get_box_edges`)

        Outputs
        -------
        dist : float
            Average point-wise distance between the two box edges
    '''
    b1_edges = get_box_edges(*box1)[:4]
    b2_edges = get_box_edges(*box2)[:4]

    # build a distance matrix between the 4 edges
    # since the order of edges may not be the same
    # for the two boxes
    dists = np.zeros((4, 4))
    for c1 in range(4):
        for c2 in range(4):
            dists[c1, c2] = get_point_distance(*b1_edges[c1], *b2_edges[c2])

    # then collapse the matrix into the minimum distance for each point
    # does not matter which axis, since we get the least distance anyway
    mindist = dists.min(axis=0)

    return np.average(mindist)


def scale_shape(params, gamma):
    '''
        scale the box by a factor of gamma
        about the center

        Inputs
        ------
        params : list
            Parameter list corresponding to the box (x, y, w, h, a).
            See `get_box_edges`
        gamma : float
            Scaling parameter. Equal to sqrt(1 - sigma), where sigma
            is the box confidence from the SHGO box-averaging step

        Outputs
        -------
        scaled_params : list
            Parameter corresponding to the box scaled by the factor gamma
    '''
    return [
        # upper left corner moves
        params[0] + (params[2] * (1 - gamma) / 2),
        params[1] + (params[3] * (1 - gamma) / 2),
        # width and height scale
        gamma * params[2],
        gamma * params[3],
        # angle does not change
        params[4]
    ]


def sigma_shape(params, sigma):
    '''
        calculate the upper and lower bounding box
        based on the sigma of the cluster

        Inputs
        ------
        params : list
            Parameter list corresponding to the box (x, y, w, h, a).
            See `get_box_edges`
        sigma : float
            Confidence of the box, given by the minimum distance of the
            SHGO box averaging step. See the `panoptes_aggregation` module.

        Outputs
        -------
        plus_sigma : list
            Parameters corresponding to the box scaled to the upper sigma bound
        minus_sigma : list
            Parameters corresponding to the box scaled to the lower sigma bound
    '''
    gamma = np.sqrt(1 - sigma)
    plus_sigma = scale_shape(params, 1 / gamma)
    minus_sigma = scale_shape(params, gamma)
    return plus_sigma, minus_sigma
