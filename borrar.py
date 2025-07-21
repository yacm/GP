import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl

def check_pdflatex():
    try:
        out = subprocess.check_output(["pdflatex", "--version"], text=True)
        print(" pdflatex está disponible:")
        print(out.splitlines()[0])
    except Exception as e:
        print(" No se pudo ejecutar pdflatex:")
        print(e)
        return False
    return True

def test_latex_plot():
    try:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
        })

        plt.plot([0, 1], [0, 1], label=r"$y = x$")
        plt.title(r"\textbf{Prueba con \LaTeX}")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.legend()
        plt.grid()
        plt.show()
        print(" LaTeX en matplotlib funcionó correctamente.")
    except Exception as e:
        print(" Error al usar LaTeX en matplotlib:")
        print(e)

# Ejecuta ambas pruebas
if check_pdflatex():
    test_latex_plot()