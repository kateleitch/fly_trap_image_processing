from __future__ import print_function
import sys
import trapcam_analysis as t

dir = sys.argv[1]
explore_cv2_parameters = sys.argv[2] # user provides a boolean
analyzer = t.TrapcamAnalyzer(dir, explore_cv2_parameters)
analyzer.run()
