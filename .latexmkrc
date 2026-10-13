# Keep the final PDF, but remove LaTeX/latexmk intermediate files after every
# successful build.  This is read automatically by `latexmk` in this project.
#
# Do not use `-C`: it also removes the generated PDF.  `-c` retains main.pdf.
$success_cmd = 'latexmk -c %T';
