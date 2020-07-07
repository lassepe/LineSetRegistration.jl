module SPLFieldModel

using GeometryBasics: Point, Line
using CoordinateTransformations: LinearMap
using StaticArrays: SMatrix

const field_len_x = 9
const field_len_y = 4
const penbox_len_x = 1
const penbox_len_y = 2
const goalbox_len_x = 2
const goalbox_len_y = 3
const center_circle_diameter = 1

function apply(tform::Transformation, l::Line)
    p1_tformed, p2_tformed = tform.(Tuple(l))
    Line(p1_tformed, p2_tformed)
end

function boxlines(boxlen_x, boxlen_y; mirror = false)
    box_lines = [
        Line(
            Point(-field_len_x / 2, -boxlen_y / 2),
            Point(-field_len_x / 2 + boxlen_x, -boxlen_y / 2),
        ),
        Line(
            Point(-field_len_x / 2, boxlen_y / 2),
            Point(-field_len_x / 2 + boxlen_x, boxlen_y / 2),
        ),
        Line(
            Point(-field_len_x / 2 + boxlen_x, -boxlen_y / 2),
            Point(-field_len_x / 2 + boxlen_x, boxlen_y / 2),
        ),
    ]

    mirror_transform = CoordinateTransformations.LinearMap(SMatrix{2,2}(diagm([-1, 1])))
    mirror ? map(l->apply(mirror_transform, l), box_lines) : box_lines
end

const field_lines = [
    # outer border
    Line(Point(-field_len_x / 2, -field_len_y / 2), Point(-field_len_x / 2, +field_len_y / 2)),
    Line(Point(-field_len_x / 2, +field_len_y / 2), Point(+field_len_x / 2, +field_len_y / 2)),
    Line(Point(+field_len_x / 2, +field_len_y / 2), Point(+field_len_x / 2, -field_len_y / 2)),
    Line(Point(+field_len_x / 2, -field_len_y / 2), Point(-field_len_x / 2, -field_len_y / 2)),
    # center line
    Line(Point(0.0, +field_len_y / 2), Point(0.0, -field_len_y / 2)),
    # penalty and goal boxes
    boxlines(penbox_len_x, penbox_len_y)...,
    boxlines(goalbox_len_x, goalbox_len_y)...,
    boxlines(penbox_len_x, penbox_len_y; mirror = true)...,
    boxlines(goalbox_len_x, goalbox_len_y; mirror = true)...,
    # # center circle
    # Circle(Point(0, 0), center_circle_diameter)
]

end
