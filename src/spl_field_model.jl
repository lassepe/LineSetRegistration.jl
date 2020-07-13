struct SPLField{L<:AbstractVector{<:Line}}
    lines::L
end

function SPLField(;
    field_len_x = 9,
    field_len_y = 4,
    penbox_len_x = 1,
    penbox_len_y = 2,
    goalbox_len_x = 2,
    goalbox_len_y = 3,
    center_circle_diameter = 1,
)

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

        if mirror
            mirror_transform = LinearMap(SMatrix{2,2}(diagm([-1, 1])))
            map(l -> transform(mirror_transform, l), box_lines)
        else
            box_lines
        end
    end

    lines = [
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

    SPLField(SizedVector{length(lines)}(lines))
end
