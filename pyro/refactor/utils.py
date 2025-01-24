import numpy


class Transformation2D:
    @staticmethod
    def translation_matrix(offset):
        """
        Compute the translation matrix.

        :param offset: The x, y offset. (2, 1)

        :return: The translation matrix. (3, 3)
        """
        return numpy.array([
            [1, 0, offset[0]],
            [0, 1, offset[1]],
            [0, 0, 1]
        ])
    
    @staticmethod
    def rotation_matrix(angle):
        """
        Compute the rotation matrix.

        :param angle: The angle of rotation.

        :return: The rotation matrix. (3, 3)
        """
        return numpy.array([
            [numpy.cos(angle), -numpy.sin(angle), 0],
            [numpy.sin(angle), numpy.cos(angle), 0],
            [0, 0, 1]
        ])
    
    @staticmethod
    def scale_matrix(scale):
        """
        Compute the scale matrix.

        :param scale: The scale factor.

        :return: The scale matrix. (3, 3)
        """
        return numpy.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def translate_rotate_matrix(translation: numpy.ndarray, angle: float) -> numpy.ndarray:
        """
        Compute the transformation matrix to translate and then rotate.

        :param translation: The translation. (2, 1)
        :param angle: The angle of rotation.

        :return: The transformation matrix. (3, 3)
        """
        return numpy.array([
            [numpy.cos(angle),  -numpy.sin(angle),  translation[0]],
            [numpy.sin(angle),  numpy.cos(angle),   translation[1]],
            [0,                 0,                  1           ]
        ])
    
    @staticmethod
    def translate_scale_matrix(scale, offset):
        """
        Compute the transformation matrix to translate and then scale.

        :param scale: The scale factor.
        :param offset: The offset. (2, 1)

        :return: The transformation matrix. (3, 3)
        """
        return numpy.array([
            [scale, 0,      offset[0]   ],
            [0,     scale,  offset[1]   ],
            [0,     0,      1           ]
        ])

    @staticmethod
    def transform_points(transformation_matrix, points):
        """
        Apply a transformation matrix to a list of points

        :param transformation_matrix: The transformation matrix. (3, 3)
        :param points: The points to transform. (n, 2)

        :return: The transformed points. (n, 2)
        """
        assert points.shape[1] == 2 or points.shape[1] == 3, "Points must be 2D or 3D"

        isnt_homogeneous = points.shape[1] == 2

        if isnt_homogeneous: # Convert to homogeneous coordinates
            points = numpy.hstack((points, numpy.ones((points.shape[0], 1))))

        return (transformation_matrix @ points.T).T[:, :2]
