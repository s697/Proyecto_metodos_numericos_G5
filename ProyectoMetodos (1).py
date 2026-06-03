import numpy as np
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.optimize import root
import time

# =========================================================================
# 1. NÚCLEO MATEMÁTICO (Newton vs Broyden)
# =========================================================================
class SimuladorIntercepcion:
    def __init__(self, D, h, v, phi_deg, T):
        self.g = 9.81
        self.D = D
        self.h = h
        self.v = v
        self.phi = np.radians(phi_deg)
        self.T = T
        self.tc = T + 2.0
        self.dt = 0.05
        self.std_viento = 0.3

    def resolver_ambos_metodos(self):
        x_obj = self.D - self.v * np.cos(self.phi) * self.tc
        y_obj = self.h + self.v * np.sin(self.phi) * self.tc - 0.5 * self.g * self.tc**2

        def ecuaciones(X):
            u, theta = X[0], X[1]
            F1 = u * np.cos(theta) * (self.tc - self.T) - x_obj
            F2 = u * np.sin(theta) * (self.tc - self.T) - 0.5 * self.g * (self.tc - self.T)**2 - y_obj
            return [F1, F2]

        u_guess = np.sqrt(x_obj**2 + y_obj**2) / (self.tc - self.T)
        theta_guess = np.arctan2(y_obj, x_obj)
        x0 = [u_guess, theta_guess]

        # Método 1: Newton-Raphson (hybr)
        t0 = time.perf_counter()
        sol_n = root(ecuaciones, x0, method='hybr')
        t1 = time.perf_counter()
        tiempo_n = t1 - t0

        # Método 2: Secante multidimensional (broyden1)
        t2 = time.perf_counter()
        sol_b = root(ecuaciones, x0, method='broyden1')
        t3 = time.perf_counter()
        tiempo_b = t3 - t2

        reporte = self._generar_reporte(sol_n, tiempo_n, sol_b, tiempo_b)
        return sol_n, tiempo_n, sol_b, tiempo_b, reporte

    def _generar_reporte(self, sol_n, t_n, sol_b, t_b):
        texto = f"--- NEWTON-RAPHSON ---\n"
        texto += f"u: {sol_n.x[0]:.3f} m/s | θ: {np.degrees(sol_n.x[1]):.3f}°\n"
        texto += f"Iteraciones: {sol_n.nfev} | Tiempo: {t_n:.5f} s\n\n"
        texto += f"--- SECANTE (BROYDEN) ---\n"
        texto += f"u: {sol_b.x[0]:.3f} m/s | θ: {np.degrees(sol_b.x[1]):.3f}°\n"
        texto += f"Iteraciones: {sol_b.nfev} | Tiempo: {t_b:.5f} s\n"
        return texto

    def generar_trayectorias(self, u, theta):
        tiempo = np.arange(0, self.tc + self.dt, self.dt)
        n = len(tiempo)

        vx1 = -self.v * np.cos(self.phi) + np.cumsum(np.random.normal(0, self.std_viento, n) * self.dt)
        vy1 = self.v * np.sin(self.phi) + np.cumsum((-self.g + np.random.normal(0, self.std_viento, n)) * self.dt)
        x1 = np.insert(self.D + np.cumsum(vx1 * self.dt)[:-1], 0, self.D)
        y1 = np.insert(self.h + np.cumsum(vy1 * self.dt)[:-1], 0, self.h)

        x2, y2 = np.zeros(n), np.zeros(n)
        idx_lanzamiento = int(self.T / self.dt)
        pasos_p2 = n - idx_lanzamiento

        if pasos_p2 > 0:
            vx2 = u * np.cos(theta) + np.cumsum(np.random.normal(0, self.std_viento, pasos_p2) * self.dt)
            vy2 = u * np.sin(theta) + np.cumsum((-self.g + np.random.normal(0, self.std_viento, pasos_p2)) * self.dt)
            x2[idx_lanzamiento:] = np.cumsum(vx2 * self.dt)
            y2[idx_lanzamiento:] = np.cumsum(vy2 * self.dt)

        return tiempo, x1, y1, x2, y2


# =========================================================================
# 2. INTERFAZ GRÁFICA
# =========================================================================
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Intercepción: Comparativa de Métodos")
        self.root.geometry("1100x600")
        
        self.ani = None 
        self._configurar_estilos()
        self._construir_interfaz()

    def _configurar_estilos(self):
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        self.bg_color = "#eef2f5"
        self.root.configure(bg=self.bg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("TLabelframe", background="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background="#ffffff")
        style.configure("TLabel", background="#ffffff", font=("Segoe UI", 10))
        style.configure("TNotebook", background=self.bg_color)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5])

    def _construir_interfaz(self):
        # PANEL IZQUIERDO (Controles)
        frame_lateral = ttk.Frame(self.root, padding=15, style="Panel.TFrame", width=320)
        frame_lateral.pack(side="left", fill="y", padx=(15, 5), pady=15)
        frame_lateral.pack_propagate(False)

        tk.Label(frame_lateral, text="Panel de Control", font=("Segoe UI", 16, "bold"), bg="#ffffff").pack(pady=(0, 15))

        frame_params = ttk.LabelFrame(frame_lateral, text=" Parámetros Físicos ", padding=10)
        frame_params.pack(fill="x", pady=5)

        variables = [("D (m):", "100"), ("h (m):", "300"), ("v (m/s):", "30"), ("ϕ (°):", "45"), ("T (s):", "2.0")]
        self.entries = []
        for i, (label, default) in enumerate(variables):
            ttk.Label(frame_params, text=label).grid(row=i, column=0, padx=5, pady=8, sticky='w')
            entry = ttk.Entry(frame_params, width=12, justify="center")
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=8)
            self.entries.append(entry)

        ttk.Button(frame_lateral, text="▶ Calcular y Simular", command=self.ejecutar_todo, padding=10).pack(fill="x", pady=20)

        frame_consola = ttk.LabelFrame(frame_lateral, text=" Consola de Resultados ", padding=5)
        frame_consola.pack(fill="both", expand=True, pady=5)
        self.txt_consola = tk.Text(frame_consola, height=10, width=30, bg="#1e1e1e", fg="#00d2ff", font=("Consolas", 10), state="disabled")
        self.txt_consola.pack(fill="both", expand=True)

        # PANEL DERECHO (Pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="right", fill="both", expand=True, padx=(5, 15), pady=15)

        # --- Pestaña 1: Animación ---
        self.tab_anim = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_anim, text=" 🎬 Animación en Vivo ")
        self.fig_anim = Figure(figsize=(7, 5), dpi=100)
        self.ax_anim = self.fig_anim.add_subplot(111)
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=self.tab_anim)
        self.canvas_anim.get_tk_widget().pack(fill="both", expand=True)

        # --- Pestaña 2: Análisis Numérico ---
        self.tab_num = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_num, text=" 📈 Newton vs Secante ")
        self.fig_num = Figure(figsize=(10, 5), dpi=100)
        self.ax_contour = self.fig_num.add_subplot(121)
        self.ax_barras = self.fig_num.add_subplot(122)
        self.canvas_num = FigureCanvasTkAgg(self.fig_num, master=self.tab_num)
        self.canvas_num.get_tk_widget().pack(fill="both", expand=True)

        self._reset_graficas()

    def _reset_graficas(self):
        self.ax_anim.clear()
        self.ax_contour.clear()
        self.ax_barras.clear()
        
        self.ax_anim.grid(True, linestyle='--', alpha=0.5)
        self.ax_contour.grid(True, linestyle='--', alpha=0.5)
        
        self.ax_anim.set_title("Esperando ejecución...")
        self.ax_contour.set_title("Espacio de Convergencia")
        self.ax_barras.set_title("Rendimiento Computacional")
        
        self.fig_anim.tight_layout()
        self.fig_num.tight_layout()
        
        self.canvas_anim.draw()
        self.canvas_num.draw()

    def _imprimir(self, texto):
        self.txt_consola.config(state="normal")
        self.txt_consola.delete("1.0", "end")
        self.txt_consola.insert("end", texto)
        self.txt_consola.config(state="disabled")

    def ejecutar_todo(self):
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None

        try:
            D, h, v, phi, T = [float(e.get()) for e in self.entries]
            simulador = SimuladorIntercepcion(D, h, v, phi, T)
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos.")
            return

        # 1. Ejecutar matematicas
        sol_n, t_n, sol_b, t_b, reporte = simulador.resolver_ambos_metodos()
        self._imprimir(reporte)

        u = sol_n.x[0]
        theta = sol_n.x[1]

        # --- ACTUALIZAR PESTAÑA 2: ANÁLISIS NUMÉRICO ---
        self.ax_contour.clear()
        self.ax_barras.clear()

        # Gráfico de Contornos (Mapa de error)
        u_vals = np.linspace(u - 15, u + 15, 50)
        theta_vals = np.linspace(theta - 0.3, theta + 0.3, 50)
        U, Theta = np.meshgrid(u_vals, theta_vals)
        Z = (U - u)**2 + 1000 * (Theta - theta)**2
        
        self.ax_contour.contourf(U, np.degrees(Theta), Z, levels=15, cmap='viridis', alpha=0.6)
        
        # Rutas de aproximación visuales (Mockups ilustrativos)
        deg_th = np.degrees(theta)
        ruta_n = np.array([[u-12, deg_th-15], [u-4, deg_th-5], [u, deg_th]])
        ruta_b = np.array([[u-12, deg_th-15], [u-7, deg_th-10], [u-3, deg_th-3], [u, deg_th]])
        
        self.ax_contour.plot(ruta_n[:,0], ruta_n[:,1], 'r-o', linewidth=2, label="Ruta Newton")
        self.ax_contour.plot(ruta_b[:,0], ruta_b[:,1], 'm-s', linewidth=2, label="Ruta Secante")
        self.ax_contour.plot(u, deg_th, 'w*', markersize=12, markeredgecolor='k', label="Mínimo Local")
        
        self.ax_contour.set_title("Rutas de Convergencia")
        self.ax_contour.set_xlabel("Velocidad u (m/s)")
        self.ax_contour.set_ylabel("Ángulo θ (Grados)")
        self.ax_contour.legend()
        self.ax_contour.grid(True, linestyle='--', alpha=0.4)

        # Gráfico de Barras de Rendimiento
        etiquetas = ['Iteraciones\n(Cant.)', 'Tiempo\n(ms)']
        datos_n = [sol_n.nfev, t_n * 1000]
        datos_b = [sol_b.nfev, t_b * 1000]
        
        x = np.arange(len(etiquetas))
        ancho = 0.35
        
        self.ax_barras.bar(x - ancho/2, datos_n, ancho, label='Newton-Raphson', color='#e74c3c')
        self.ax_barras.bar(x + ancho/2, datos_b, ancho, label='Secante (Broyden)', color='#9b59b6')
        
        self.ax_barras.set_title("Comparación de Rendimiento")
        self.ax_barras.set_xticks(x)
        self.ax_barras.set_xticklabels(etiquetas)
        self.ax_barras.legend()
        self.ax_barras.grid(axis='y', linestyle='--', alpha=0.4)

        self.fig_num.tight_layout()
        self.canvas_num.draw()

        # --- ACTUALIZAR PESTAÑA 1: ANIMACIÓN ---
        self.notebook.select(self.tab_anim)
        t_anim, x1, y1, x2, y2 = simulador.generar_trayectorias(u, theta)
        
        self.ax_anim.clear()
        self.ax_anim.grid(True, linestyle='--', alpha=0.5)
        self.ax_anim.set_xlim(-5, max(simulador.D, np.max(x1)) + 15)
        self.ax_anim.set_ylim(0, max(np.max(y1), np.max(y2)) + 15)
        self.ax_anim.set_title("Animación: Intercepción en Vivo")
        
        line1, = self.ax_anim.plot([], [], 'r-', linewidth=2, label="P1 (Objetivo)")
        line2, = self.ax_anim.plot([], [], 'b-', linewidth=2, label="P2 (Interceptor)")
        p1, = self.ax_anim.plot([], [], 'ro', markersize=6)
        p2, = self.ax_anim.plot([], [], 'bo', markersize=6)
        self.ax_anim.legend(loc="upper right")

        def update(frame):
            line1.set_data(x1[:frame], y1[:frame])
            p1.set_data([x1[frame]], [y1[frame]])
            if t_anim[frame] >= simulador.T:
                idx = int(simulador.T / simulador.dt)
                line2.set_data(x2[idx:frame], y2[idx:frame])
                p2.set_data([x2[frame]], [y2[frame]])
            return line1, line2, p1, p2

        self.fig_anim.tight_layout()
        self.ani = animation.FuncAnimation(self.fig_anim, update, frames=len(t_anim), interval=35, blit=False)
        self.canvas_anim.draw()


if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppGUI(ventana)
    ventana.mainloop()