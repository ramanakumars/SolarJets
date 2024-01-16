import numpy as np
import os
from dataclasses import dataclass
from shapely.geometry import Polygon
from panoptes_aggregation.reducers.shape_metric_IoU import IoU_metric
from panoptes_aggregation.reducers.point_process_data import temporal_metric
import yaml


with open(os.path.join(os.path.split(__file__)[0], '..',
                       'configs/Reducer_config_workflow_21225_V50.59_shapeExtractor_temporalRotateRectangle.yaml'), 'r') as infile:
    EPS_T = yaml.safe_load(infile)['reducer_config']['shape_reducer_dbscan']['eps_t']


@dataclass
class BasePoint:
    x: float
    y: float
    displayTime: float
    subject_id: int
    probability: float = 0

    def to_dict(self):
        data = {}
        data['subject_id'] = self.subject_id
        data['x'] = self.x
        data['y'] = self.y
        data['displayTime'] = self.displayTime
        data['probability'] = self.probability

        if hasattr(self, 'extracts'):
            data['extracts'] = [ext.to_dict() for ext in self.extracts]

        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(x=data['x'], y=data['y'], displayTime=data['displayTime'], subject_id=data['subject_id'], probability=data['probability'])

        if 'extracts' in data:
            obj.extracts = []
            for extract in data['extracts']:
                ext = cls(x=extract['x'], y=extract['y'], displayTime=extract['displayTime'], subject_id=extract['subject_id'], probability=extract['probability'])
                obj.extracts.append(ext)

        return obj

    @property
    def coordinate(self):
        return np.asarray([self.x, self.y])

    @property
    def extract_dists(self):
        if not hasattr(self, '_extract_dists'):
            if not hasattr(self, 'extracts'):
                raise ValueError('Point does not have extracts!')
            dists = []
            for extract in self.extracts:
                dists.append(get_point_distance(self, extract))
            self._extract_dists = np.mean(dists)

        return self._extract_dists


@dataclass
class Box:
    xcenter: float
    ycenter: float
    width: float
    height: float
    angle: float
    displayTime: float
    subject_id: int
    probability: float = 0

    def to_dict(self):
        data = {}
        data['subject_id'] = self.subject_id
        data['xcenter'] = self.xcenter
        data['ycenter'] = self.ycenter
        data['width'] = self.width
        data['height'] = self.height
        data['angle'] = self.angle
        data['displayTime'] = self.displayTime
        data['probability'] = self.probability

        if hasattr(self, 'extracts'):
            data['extracts'] = [ext.to_dict() for ext in self.extracts]

        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(xcenter=data['xcenter'], ycenter=data['ycenter'], width=data['width'], height=data['height'],
                  angle=data['angle'], displayTime=data['displayTime'], subject_id=data['subject_id'], probability=data['probability'])

        if 'extracts' in data:
            obj.extracts = []
            for extract in data['extracts']:
                ext = cls(xcenter=extract['xcenter'], ycenter=extract['ycenter'], width=extract['width'], height=extract['height'],
                          angle=extract['angle'], displayTime=extract['displayTime'], subject_id=extract['subject_id'], probability=extract['probability'])
                obj.extracts.append(ext)

        return obj

    def get_box_edges(self):
        '''
            Return the corners of the box given one corner, width, height
            and angle

            Outputs
            --------
            corners : numpy.ndarray
                Length 4 array with coordinates of the box edges
        '''
        centre = np.array([self.xcenter, self.ycenter])
        original_points = np.array(
            [
                [self.xcenter - 0.5 * self.width, self.ycenter - 0.5 * self.height],  # This would be the box if theta = 0
                [self.xcenter + 0.5 * self.width, self.ycenter - 0.5 * self.height],
                [self.xcenter + 0.5 * self.width, self.ycenter + 0.5 * self.height],
                [self.xcenter - 0.5 * self.width, self.ycenter + 0.5 * self.height],
                # repeat the first point to close the loop
                [self.xcenter - 0.5 * self.width, self.ycenter - 0.5 * self.height]
            ]
        )
        rotation = np.array([[np.cos(self.angle), np.sin(self.angle)], [-np.sin(self.angle), np.cos(self.angle)]])
        corners = np.matmul(original_points - centre, rotation) + centre
        return corners

    def get_shapely_polygon(self):
        return Polygon(self.get_box_edges())

    def get_plus_minus_sigma(self, sigma):
        # calculate the bounding box for the cluster confidence
        plus_sigma, minus_sigma = sigma_shape(
            [self.xcenter, self.ycenter, self.width, self.height, self.angle], sigma)

        plus_box = Box(*plus_sigma, self.displayTime, self.subject_id)
        minus_box = Box(*minus_sigma, self.displayTime, self.subject_id)

        return plus_box.get_box_edges(), minus_box.get_box_edges()

    @property
    def params(self):
        return [self.xcenter, self.ycenter, self.width, self.height, self.angle, self.displayTime]

    @property
    def extract_IoU(self):
        if not hasattr(self, '_extract_IoU'):
            if not hasattr(self, 'extracts'):
                raise ValueError('Box does not have extracts!')
            IoUs = []
            for extract in self.extracts:
                IoUs.append(1. - IoU_metric([self.xcenter, self.ycenter, self.width, self.height, self.angle, self.displayTime],
                                            [extract.xcenter, extract.ycenter, extract.width, extract.height, extract.angle, extract.displayTime],
                                            'temporalRotateRectangle', EPS_T))
            self._extract_IoU = np.mean(IoUs)

        return self._extract_IoU


def get_point_distance(p1: BasePoint, p2: BasePoint) -> float:
    '''
        Get Euclidiean distance between two points p1 and p2

        Inputs
        ------
        p1: BasePoint
            BasePoint object for the first point
        p2: BasePoint
            BasePoint object for the second point

        Outputs
        --------
        dist : float
            Euclidian distance between (x0, y0) and (x1, y1)
    '''
    return temporal_metric([p1.x, p1.y, p1.displayTime], [p2.x, p2.y, p2.displayTime])


def get_box_distance(box1: Box, box2: Box) -> float:
    '''
        Get point-wise distance betweeen 2 boxes.
        Calculates and find the average distance between each edge
        for each box

        Inputs
        ------
        box1 : Box
            parameters corresponding to the first box (see `get_box_edges`)
        box2 : Box
            parameters corresponding to the second box (see `get_box_edges`)

        Outputs
        -------
        dist : float
            Average point-wise distance between the two box edges
    '''
    b1_edges = box1.get_box_edges()[:4]
    b2_edges = box2.get_box_edges()[:4]

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


def get_box_iou(box1: Box, box2: Box) -> float:
    return 1 - IoU_metric(box1.params, box2.params, 'temporalRotateRectangle', EPS_T)


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
