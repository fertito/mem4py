python.exe .\wing_fsi_cfd.py --wing ..\..\mem4py_working_dir\singleskin\ --iteration 0 --open
python.exe .\paraglider_fsi.py --wing ..\..\mem4py_working_dir\singleskin\ --iteration 0 --solve --pretension
python.exe .\wing_fsi_cfd.py --wing ..\..\mem4py_working_dir\singleskin\ --iteration 1 --open
python.exe .\paraglider_fsi.py --wing ..\..\mem4py_working_dir\singleskin\ --iteration 1 --solve --pretension
python.exe .\wing_fsi_cfd.py --wing ..\..\mem4py_working_dir\singleskin\ --iteration 2 --open
python.exe .\paraglider_fsi.py --wing ..\..\mem4py_working_dir\singleskin\ --iteration 2 --solve --pretension
