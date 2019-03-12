from __future__ import print_function
import sys
import trapcam_analysis as t

dir = sys.argv[1]
analyzer = t.TrapcamAnalyzer(dir)
analyzer.run()


# NEED TO THINK OF BEST WAY TO SAVE DATA
