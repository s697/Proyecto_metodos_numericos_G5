import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import messagebox
from scipy.optimize import root
import time
import tracemalloc

# Constante de gravedad
g = 9.81

# --- 1. MÉTODOS NUMÉRICOS ---
def resolver_parametros_intercepcion(D, h, v, phi_deg, T, tc):
    phi = np.radians(phi_deg)

    # Posición objetivo en el tiempo tc
    x_obj = D - v * np.cos(phi) * tc
    y_obj = h + v * np.sin(phi) * tc - 0.5 * g * tc**2

    # Sistema de ecuaciones
    def ecuaciones(X):
        u, theta = X[0], X[1]

        F1 = u * np.cos(theta) * (tc - T) - x_obj

        F2 = (
            u * np.sin(theta) * (tc - T)
            - 0.5 * g * (tc - T)**2
            - y_obj
        )

        return [F1, F2]

    # Estimación inicial
    u_guess = np.sqrt(x_obj**2 + y_obj**2) / (tc - T)
    theta_guess = np.arctan2(y_obj, x_obj)

    x0 = [u_guess, theta_guess]

    # ==========================================
    # INICIO MEDICIÓN DE RECURSOS
    # ==========================================

    tracemalloc.start()

    # -------- NEWTON (HYBR) --------

    inicio_newton = time.perf_counter()

    sol_newton = root(
        ecuaciones,
        x0,
        method='hybr'
    )

    fin_newton = time.perf_counter()

    tiempo_newton = fin_newton - inicio_newton

    # -------- BROYDEN --------

    inicio_broyden = time.perf_counter()

    sol_broyden = root(
        ecuaciones,
        x0,
        method='broyden1'
    )

    fin_broyden = time.perf_counter()

    tiempo_broyden = fin_broyden - inicio_broyden

    memoria_actual, memoria_pico = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # ==========================================
    # RESULTADOS NUMÉRICOS
    # ==========================================

    print("\n")
    print("=" * 60)
    print("COMPARACIÓN DE MÉTODOS NUMÉRICOS")
    print("=" * 60)

    print("\nNewton-Raphson (HYBR)")
    print(f"u = {sol_newton.x[0]:.4f} m/s")
    print(f"θ = {np.degrees(sol_newton.x[1]):.4f}°")
    print(f"Convergencia = {sol_newton.success}")
    print(f"Evaluaciones = {sol_newton.nfev}")
    print(f"Tiempo = {tiempo_newton:.8f} s")

    print("\nBroyden")
    print(f"u = {sol_broyden.x[0]:.4f} m/s")
    print(f"θ = {np.degrees(sol_broyden.x[1]):.4f}°")
    print(f"Convergencia = {sol_broyden.success}")
    print(f"Evaluaciones = {sol_broyden.nfev}")
    print(f"Tiempo = {tiempo_broyden:.8f} s")

    print("\n")
    print("=" * 60)
    print("RECURSOS COMPUTACIONALES")
    print("=" * 60)

    print(f"Memoria actual: {memoria_actual / 1024:.2f} KB")
    print(f"Memoria pico:   {memoria_pico / 1024:.2f} KB")

    print("\nComplejidad Temporal:")
    print("  Resolución del sistema no lineal: O(k)")
    print("  Simulación de Euler: O(n)")

    print("\nComplejidad Espacial:")
    print("  Almacenamiento de trayectorias: O(n)")

    print("\nComplejidad Total:")
    print("  O(n) + O(k) ≈ O(n)")

    print("=" * 60)

    return (
        sol_newton.x[0],
        np.degrees(sol_newton.x[1])
    )
# --- 2. SIMULACIÓN ESTOCÁSTICA Y ANIMACIÓN ---
def simular_y_animar(D, h, v, phi, T, u, theta, tc):
    dt = 0.05
    tiempo = np.arange(0, tc + dt, dt)

    # Ruido blanco (Viento) - Desviación estándar de la velocidad
    std_viento = 0.3

    # Arrays de posición
    x1, y1 = np.zeros(len(tiempo)), np.zeros(len(tiempo))
    x2, y2 = np.zeros(len(tiempo)), np.zeros(len(tiempo))

    # Condiciones iniciales P1
    x1[0], y1[0] = D, h
    vx1, vy1 = -v * np.cos(np.radians(phi)), v * np.sin(np.radians(phi))

    # Condiciones iniciales P2
    vx2, vy2 = 0, 0
    lanzado = False

    # Integración de Euler con Ruido Blanco
    for i in range(1, len(tiempo)):
        t = tiempo[i]

        # P1 con viento (ruido)
        vx1 += np.random.normal(0, std_viento) * dt
        vy1 += (-g + np.random.normal(0, std_viento)) * dt
        x1[i] = x1[i-1] + vx1 * dt
        y1[i] = y1[i-1] + vy1 * dt

        # P2 lógica de lanzamiento y viento
        if t >= T:
            if not lanzado:
                # Lanzamiento
                x2[i-1], y2[i-1] = 0, 0
                vx2 = u * np.cos(np.radians(theta))
                vy2 = u * np.sin(np.radians(theta))
                lanzado = True

            vx2 += np.random.normal(0, std_viento) * dt
            vy2 += (-g + np.random.normal(0, std_viento)) * dt
            x2[i] = x2[i-1] + vx2 * dt
            y2[i] = y2[i-1] + vy2 * dt
        else:
            x2[i], y2[i] = 0, 0 # Aún en base

    # Configuración de Animación
    fig, ax = plt.subplots()
    ax.set_xlim(-5, D + 5)
    ax.set_ylim(0, max(np.max(y1), np.max(y2)) + 10)
    ax.set_title("Simulación de Intercepción con Viento (Ruido Blanco)")
    ax.set_xlabel("Distancia (x)")
    ax.set_ylabel("Altura (y)")

    line1, = ax.plot([], [], 'r-', label="Proyectil 1")
    line2, = ax.plot([], [], 'b-', label="Proyectil 2")
    punto1, = ax.plot([], [], 'ro')
    punto2, = ax.plot([], [], 'bo')
    ax.legend()

    def update(frame):
        line1.set_data(x1[:frame], y1[:frame])
        punto1.set_data([x1[frame]], [y1[frame]])
        if tiempo[frame] >= T:
            line2.set_data(x2[int(T/dt):frame], y2[int(T/dt):frame])
            punto2.set_data([x2[frame]], [y2[frame]])
        return line1, line2, punto1, punto2

    ani = animation.FuncAnimation(fig, update, frames=len(tiempo), interval=50, blit=True)
    plt.show()

# --- 3. INTERFAZ GRÁFICA (GUI) ---
def iniciar_gui():
    root_tk = tk.Tk()
    root_tk.title("Configuración de Proyectiles")

    labels = ['D (Posición X inicial):', 'h (Altura inicial):', 'v (Velocidad inicial P1):',
              'ϕ (Ángulo P1 en grados):', 'T (Tiempo espera P2):']
    defaults = ['100', '50', '30', '45', '2.0']
    entries = []

    for i, text in enumerate(labels):
        tk.Label(root_tk, text=text).grid(row=i, column=0, padx=10, pady=5, sticky='e')
        entry = tk.Entry(root_tk)
        entry.insert(0, defaults[i])
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries.append(entry)

    def on_simular():
        try:
            D = float(entries[0].get())
            h = float(entries[1].get())
            v = float(entries[2].get())
            phi = float(entries[3].get())
            T = float(entries[4].get())

            # Definimos el tiempo de colisión heurísticamente (ej. 2 segundos después de lanzar P2)
            tc = T + 2.0

            u, theta = resolver_parametros_intercepcion(D, h, v, phi, T, tc)

            # Mostrar resultados en GUI antes de animar
            messagebox.showinfo("Resultados", f"Parámetros P2 Encontrados:\n\nVelocidad (u): {u:.2f} m/s\nÁngulo (θ): {theta:.2f}°\n\nPresione OK para ver la simulación.")

            simular_y_animar(D, h, v, phi, T, u, theta, tc)
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos.")

    tk.Button(root_tk, text="Simular", command=on_simular, bg='lightblue').grid(row=5, column=0, columnspan=2, pady=20)
    root_tk.mainloop()

if __name__ == "__main__":
    iniciar_gui()