@echo off
echo this program will profile where there is a decorator @profile placed above a function.
echo enter the python file to profile
set /p "filename=	"
kernprof -l -v %filename%
echo python -m line_profiler %filename%.lprof
pause