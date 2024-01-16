import numpy as np
from dateutil.parser import parse
from .shape_utils import BasePoint, Box
from dataclasses import dataclass
import datetime


@dataclass
class Jet:
    '''
        Oject to hold the data associated with a single jet.
        Contains the start/end positions and associated extracts,
        and the box (as a `shapely.Polygon` object) and corresponding
        extracts
    '''

    subject_id: int
    sol_standard: str
    start: BasePoint
    end: BasePoint
    box: Box

    def __post_init__(self):
        self.autorotate()
        return

    def to_dict(self):
        data = {}
        data['subject_id'] = self.subject_id
        data['sol_standard'] = self.sol_standard
        data['start'] = self.start.to_dict()
        data['end'] = self.end.to_dict()
        data['box'] = self.box.to_dict()
        if hasattr(self, 'time_info'):
            data['time_info'] = self.time_info

        return data

    @classmethod
    def from_dict(cls, data):
        start = BasePoint.from_dict(data['start'])
        end = BasePoint.from_dict(data['end'])
        box = Box.from_dict(data['box'])

        obj = cls(subject_id=data['subject_id'], sol_standard=data['sol_standard'], start=start, end=end, box=box)

        if 'time_info' in data:
            obj.add_time(parse(data['time_info']['subject_start_time']), parse(data['time_info']['subject_end_time']))

        return obj

    def add_time(self, start_time, end_time):
        if isinstance(start_time, np.ndarray):
            start_time = start_time.item()
        if isinstance(end_time, np.ndarray):
            end_time = end_time.item()
        dt = (end_time - start_time).total_seconds()
        self.time_info = {}
        self.time_info['subject_start_time'] = start_time
        self.time_info['subject_end_time'] = end_time
        self.time_info['subject_dt'] = dt

        self.time_info['box'] = start_time + datetime.timedelta(seconds=self.box.displayTime * dt)
        self.time_info['start'] = start_time + datetime.timedelta(seconds=self.start.displayTime * dt)
        self.time_info['end'] = start_time + datetime.timedelta(seconds=self.end.displayTime * dt)

    def get_extract_starts(self):
        '''
            Get the extract coordinates associated with the
            starting base points

            Outputs
            -------
            coords : numpy.ndarray
                coordinates of the base points from the extracts
                for the start frame
        '''
        return np.asarray([p.coordinates for p in self.start_extracts])

    def get_extract_ends(self):
        '''
            Get the extract coordinates associated with the
            final base points

            Outputs
            -------
            coords : numpy.ndarray
                coordinates of the base points from the extracts
                for the final frame
        '''

        return np.asarray([p.coordinates for p in self.end_extracts])

    def get_extract_boxes(self):
        '''
            Get the extract shapely Polygons corresponding
            to the boxes

            Outputs
            -------
            boxes : list
                List of `shapely.Polygon` objects corresponding
                to individual boxes in the extracts
        '''
        return [box.get_polygon() for box in self.box_extracts]

    def plot(self, ax, plot_sigma=True):
        '''
            Plot the the data for this jet object. Plots the
            start and end clustered points, and the associated
            extracts. Also plots the clustered and extracted
            box. Also plots a vector from the base to the top of the box

            Input
            -----
            ax : `matplotlib.Axes()`
                axis object to plot onto. The general use case for this
                function is to plot onto an existing axis which already has
                the subjet image and potentially other jet plots

            Outputs
            -------
            ims : list
                list of `matplotlib.Artist` objects that was created for this plot
        '''
        boxplot, = ax.plot(*self.box.exterior.xy, 'b-',
                           linewidth=0.8, zorder=10)
        startplot, = ax.plot(*self.start, color='limegreen',
                             marker='x', markersize=2, zorder=10)
        endplot, = ax.plot(*self.end, 'rx', markersize=2, zorder=10)

        start_ext = self.get_extract_starts()
        end_ext = self.get_extract_ends()

        # plot the extracts (start, end, box)
        startextplot, = ax.plot(
            start_ext[:, 0], start_ext[:, 1], 'k.', markersize=1.)
        endextplot, = ax.plot(
            end_ext[:, 0], end_ext[:, 1], 'k.', markersize=1.)
        boxextplots = []
        for box in self.get_extract_boxes():
            iou = box.intersection(self.box).area / box.union(self.box).area
            boxextplots.append(
                ax.plot(*box.exterior.xy, 'k-', linewidth=0.5, alpha=0.65 * iou + 0.05)[0])

        # find the center of the box, so we can draw a vector through it
        center = np.mean(self.box.get_box_edges()[:, :4], axis=1)

        # create the rotation matrix to rotate a vector from solar north tos
        # the direction of the jet
        rotation = np.asarray([[np.cos(self.angle), -np.sin(self.angle)],
                               [np.sin(self.angle), np.cos(self.angle)]])

        # create a vector by choosing the top of the jet and base of the jet
        # as the two points
        point0 = center + np.matmul(rotation, np.asarray([0, self.height / 2.]))
        point1 = center + np.matmul(rotation, np.asarray([0, -self.height / 2.]))
        vec = point1 - point0

        base_points, height_points = self.get_width_height_pairs()

        arrowplot = ax.arrow(
            *point0, vec[0], vec[1], color='white', width=2, length_includes_head=True, head_width=10)

        return [boxplot, startplot, endplot, startextplot, endextplot, *boxextplots, arrowplot]

    def autorotate(self):
        '''
            Find the rotation of the jet wrt to solar north and
            find the base width and height of the box
        '''
        box_points = self.box.get_box_edges()[:4, :]

        # find the distance between each point and the starting base
        dists = [np.linalg.norm((point - self.start.coordinate)) for point in box_points]
        sorted_dists = np.argsort(dists)

        # the base points are the two points closest to the start
        base_points = np.array([box_points[sorted_dists[0]], box_points[sorted_dists[1]]])

        # the height points are the next two
        rolled_points = np.delete(np.roll(box_points, -sorted_dists[0],
                                          axis=0), 0, axis=0)

        # we want to make sure that the order of the points
        # is in such a way that the point closest to the base
        # comes first -- this will ensure that the next point is
        # along the height line
        if np.linalg.norm(rolled_points[0, :] - base_points[1, :]) == 0:
            height_points = rolled_points[:2]
        else:
            height_points = rolled_points[::-1][:2]

        # also figure out the angle and additional size metadata
        # the angle is the angle between the height points and the base
        dh = height_points[1] - height_points[0]
        self.box.angle = np.arctan2(dh[0], -dh[1])
        self.box.width = np.linalg.norm(base_points[1] - base_points[0])
        self.box.height = np.linalg.norm(dh)

    def get_width_height_pairs(self):
        '''
            Outputs the base points and the height line segment
            points

            Outputs
            -------
            base_points : `numpy.ndarray`
                the pair of points that correspond to the base of the jet
            height_points : `numpy.ndarray`
                the pair of points that correspond to the height of the jet
        '''

        return self.base_points, self.height_points
