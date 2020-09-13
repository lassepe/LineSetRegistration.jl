import LineSetRegistration
import GeometryBasics
import BSON

BSON.@load "../debug_data/test_lines.bson" test_lines
LineSetRegistration.run_test(test_lines)
