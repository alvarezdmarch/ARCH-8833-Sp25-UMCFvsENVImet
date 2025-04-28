@echo off
cd C:
cd C:\ENVImet5\win64
if errorlevel 1 goto :failed
"C:\ENVImet5\win64\envicore_console.exe" "C:\Users\makel\AppData\Roaming\ENVI-met\Workspace\SimpleCaseIncreasedBoundary06" "Project" "SimpleCaseIncreasedBoundarySimulationWithClouds.simx"
: failed
echo If Envimet is not in default unit 'C:' connect installation folder.
pause
