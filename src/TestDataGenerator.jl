module LineTestDataGenerator

using CoordinateTransformations: Translation
using Random: shuffle
include("SPLFieldModel.jl")
using SPLFieldModel: SPLField

"Artificial noise on lines."
function noisify(line; max_n_segments = 3, rng = Random.GLOBAL_RNG, σx = 0.01, σy = 0.01)
    n_segments = rand(1:max_n_segments)
    line_vec, p1, p2 = line_vector(line)
    line_segement_thresholds = vcat(0.0, sort(rand(rng, n_segments - 1)), 1.0)
    rand_translation(rng) = Translation(σx * randn(rng), σy * randn(rng))
    map(line_segement_thresholds, vcat(line_segement_thresholds[2:end])) do t_start, t_end
        p_start = rand_translation(rng)(p1 + t_start * line_vec)
        p_end = rand_translation(rng)(p1 + t_end * line_vec)
        Line(p_start, p_end)
    end
end

"The known lines on the map."
spl_field = SPLField()
"Some perceived lines."

"""
Sample up to `max_n_lines` from `spl_field.lines` (without replacement) and apply a random
`rigid_transformation` and a artificial noise to each line.
"""
function generate_test_lines(
    max_n_lines = 10;
    spl_field = spl_field,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)
    lines = Iterators.take(shuffle(rng, spl_field.lines), rand(rng, 1:max_n_lines))
    rigid_tform = pose_transformation(randn(rng, 3))
    mapreduce(vcat, lines) do l
        noisify(rigid_tform(l); rng, kwargs...)
    end
end

end # module
