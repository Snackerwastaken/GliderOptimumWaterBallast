import numpy as np
import pandas as pd
from math import sqrt, radians, cos
import warnings
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)
import streamlit as st

st.markdown('<h1 style="text-align: center;">Berechnung des optimalen Wasserballasts</h2>', unsafe_allow_html=True)

flugzeuge = {
    "LS4": {
        "G_a_min": 290,
        "G_a_max": 525,
        "S": 10.5,
        "G_p": 338.1,
        "Xone": 90, "Yone": -0.62,
        "Xtwo": 120, "Ytwo": -0.9,
        "Xthree": 150, "Ythree": -1.46
    },
    "LS8": {
        "G_a_min": 345,
        "G_a_max": 525,
        "S": 10.5,
        "G_p": 346.5,
        "Xone": 83, "Yone": -0.59,
        "Xtwo": 120, "Ytwo": -0.7,
        "Xthree": 150, "Ythree": -1.65
    },

    "ASW19B": {
        "G_a_min": 330,
        "G_a_max": 408,
        "S": 11,
        "G_p": 370.8,
        "Xone": 80, "Yone": -0.73,
        "Xtwo": 120, "Ytwo": -0.98,
        "Xthree": 150, "Ythree": -1.49
    },

    "LS1f": {
        "G_a_min": 300,
        "G_a_max": 390,
        "S": 9.75,
        "G_p":390 ,
        "Xone": 80, "Yone": -0.75,
        "Xtwo": 120, "Ytwo": -0.97,
        "Xthree": 150, "Ythree": -1.49
    },
    "ASG29 - 18m": {
        "G_a_min": 350,
        "G_a_max": 600,
        "S":10.5,
        "G_p": 450,
        "Xone": 90, "Yone": -0.48,
        "Xtwo": 120, "Ytwo": -0.67,
        "Xthree": 150, "Ythree": -1.09
    },
    "LS7": {
        "G_a_min": 300,
        "G_a_max": 486,
        "S":9.73,
        "G_p": 347.5,
        "Xone": 90, "Yone": -0.7,
        "Xtwo": 120, "Ytwo": -0.93,
        "Xthree": 150, "Ythree": -1.5
    },
    "Ventus 3 - 18m": {
        "G_a_min": 380,
        "G_a_max": 600,
        "S":10.84,
        "G_p": 477,
        "Xone": 90, "Yone": -0.53,
        "Xtwo": 120, "Ytwo": -0.61,
        "Xthree": 150, "Ythree": -0.94
    },
    "Duo Discus XLT": {
        "G_a_min": 410,
        "G_a_max": 750,
        "S":16.4,
        "G_p":700 ,
        "Xone": 95, "Yone": -0.6,
        "Xtwo": 120, "Ytwo": -0.75,
        "Xthree": 150, "Ythree": -1.15
    }

}


wahl = st.selectbox(
    "Flugzeug auswählen",
    options=["...", "Duo Discus XLT", "LS4", "LS8", "ASW19B", "LS1f","LS7","Ventus 3 - 18m", "ASG29 - 18m", "Eigener Flieger"],
    index=0
)

# Nichts ausführen, solange noch kein Flugzeug gewählt wurde
if wahl == "...":
    pass 
else:
    # Flugzeugdaten übernehmen oder eigenen Flieger eingeben
    if wahl == "Eigener Flieger":
        st.subheader("Eigene Flugzeugdaten eingeben")
        G_a_min = st.number_input("Minimales Abfluggewicht [kg]", value=300.0, step=1.0, format="%.1f")
        G_a_max = st.number_input("Maximales Abfluggewicht [kg]", value=500.0, step=1.0, format="%.1f")
        S = st.number_input("Flügelfläche [m²]", value=10.5, step=0.1, format="%.1f")
        

        st.subheader("Gleitpolare Punkte")
        G_p = st.number_input("Referenzgewicht Flugzeugpolare [kg]", value=340.0, step=1.0, format="%.1f")
        Xone = st.number_input("X1 [km/h]", value=90.0, step=1.0, format="%.1f")
        Yone = st.number_input("Y1 [m/s]", value=-0.6, step=0.01, format="%.2f")
        Xtwo = st.number_input("X2 [km/h]", value=120.0, step=1.0, format="%.1f")
        Ytwo = st.number_input("Y2 [m/s]", value=-0.9, step=0.01, format="%.2f")
        Xthree = st.number_input("X3 [km/h]", value=150.0, step=1.0, format="%.1f")
        Ythree = st.number_input("Y3 [m/s]", value=-1.4, step=0.01, format="%.2f")
    else:
        params = flugzeuge[wahl]
        G_a_min = params["G_a_min"]
        G_a_max = params["G_a_max"]
        S = params["S"]
        G_p = params["G_p"]
        Xone, Yone = params["Xone"], params["Yone"]
        Xtwo, Ytwo = params["Xtwo"], params["Ytwo"]
        Xthree, Ythree = params["Xthree"], params["Ythree"]




    # --- Eingaben / Konstanten ---
    D = st.slider(
        "Bitte PFD[km] eingeben",
        min_value=100,     # Minimalwert
        max_value=1300,    # Maximalwert
        value=300,         # Startwert
        step=50,          # Schrittweite
        format="%.0f"      # Anzeige mit 1 Nachkommastelle
    )
    #st.write(f"Eingegebener PFD: {D}")
    D= D * 1000
    #D = 300 * 1000  # Überlandflugdistanz [m]





    Thermik = st.slider(
        "Bitte den zu erwartenden Thermikdurchschnitt [m/s] eingeben",
        min_value=1.00,     # Minimalwert
        max_value=5.00,    # Maximalwert
        value=2.00,         # Startwert
        step=0.25,          # Schrittweite
        format="%.2f"      # Anzeige mit 1 Nachkommastelle
    )

    wolken = st.slider(
        "Wolkenstraßen in %",
        min_value=0,     # Minimalwert
        max_value=60,    # Maximalwert
        value=30,         # Startwert
        step=5,          # Schrittweite
        format="%.0f"      # Anzeige mit 1 Nachkommastelle
    )

    anteile = np.array([10, 20, 20, 20], dtype=float )
    gesamt = np.sum(anteile) + wolken
    anteile_skaliert = anteile / np.sum(anteile) * (100 - wolken)
    frac = np.append(anteile_skaliert, wolken) 

    
    # Thermikmodell
    thermik = {
        "a": np.array([Thermik,   Thermik*1.4, Thermik*1.6, Thermik*1.9, 1.0]),            # [m/s]
        "b": np.array([-0.00005, -0.00008, -0.00009, -0.0001, 0.0]),
        "frac": np.array([frac[0], frac[1], frac[2], frac[3], wolken]) / 100.0,
        "type": ['A1', 'A2', 'B1', 'B2', 'GL']
    }

    ballern= Thermik*0.5#(2/3)
    
    # Radius-Vektor (in m)
    r = np.arange(30, 240)  # 30 .. 239

    # Umgebung
    rho = 1
    g = 9.81

    # Konstanten vorbereiten
    G_a_list = np.arange(G_a_min, G_a_max + 1, 5)
    n_G = len(G_a_list)

    # Ergebnis-Container (pre-allocate arrays)
    max_w_ST = np.zeros((5, n_G))
    x0 = np.zeros((5, n_G))
    y0 = np.zeros((5, n_G))
    x_tan = np.zeros((5, n_G))
    y_tan = np.zeros((5, n_G))
    GZ = np.zeros((5, n_G))
    r_max = np.zeros((5, n_G))
    phi_max = np.zeros((5, n_G))
    t_X = np.zeros((5, n_G))
    x_Gl = np.zeros((1, n_G))
    y_Gl = np.zeros((1, n_G))

    # Hilfsfunktion: berechne_w_ST (PLATZHALTER!)
    def berechne_w_ST(C_A_K, C_W_K, g, r_array, w_A_func, G, konst):
        w_ST = np.zeros_like(r_array, dtype=float)
        w_A = np.zeros_like(r_array, dtype=float)
        w_SK = np.zeros_like(r_array, dtype=float)
        phi = np.zeros_like(r_array, dtype=float)

        # Startwert phi (deg -> rad)
        phi_deg = 45

        for i, r in enumerate(r_array):
            phi_i = np.deg2rad(phi_deg)

            # Iteration zur Berechnung von phi_i
            for _ in range(1000):
                V_K = konst * np.sqrt(1.0 / (C_A_K * np.cos(phi_i)))
                phi_new = np.arctan((V_K**2) / (g * r))
                if abs(phi_new - phi_i) < 1e-8:
                    break
                phi_i = phi_new

            phi[i] = np.rad2deg(phi_i)

            # w_SK berechnen
            w_SK[i] = konst * (C_W_K / (C_A_K**1.5 * (np.cos(phi_i)**1.5)))

            # w_A aus Funktion
            w_A[i] = w_A_func(r)

            # Gesamtsteiggeschwindigkeit
            w_ST[i] = w_A[i] - w_SK[i]

        return w_ST, w_A, w_SK, phi

    # Berechnung Koeffizienten der Parabel (Polare) - analog MATLAB
    a_old = ( (Ythree - Ytwo) / ((Xthree - Xtwo)*(Xthree - Xone)) ) - ( (Yone - Ytwo) / ((Xone - Xtwo)*(Xthree - Xone)) )
    b_old = (Yone - Ytwo + a_old*(Xtwo**2 - Xone**2)) / (Xone - Xtwo)
    c_old = Yone - a_old*(Xone**2) - b_old*Xone

    # Hauptschleife über Gewichte
    V_XC = np.zeros(n_G)
    thermikdurch = np.zeros(n_G)

    for ui, Ga in enumerate(G_a_list):
        G = Ga * g
        konst = np.sqrt(2 * G / (rho * S))

        # Flächenbelastungen (nur zur Information)
        Fb_p = G_p / S
        Fb_a = Ga / S
        R = Ga / G_p

        # Polarenkoeffizienten skaliert
        a = a_old / np.sqrt(R)
        b = b_old
        c = c_old * np.sqrt(R)

        # Polare-Funktion
        def f_polar(x):
            return a * x**2 + b * x + c

        x = np.linspace(60, 250, 1000)
        y = f_polar(x)

        # Schwerpunktpolare
        # x0 = -b/(2a)
        if a == 0:
            x0_val = -np.inf
            y0_val = np.nan
        else:
            x0_val = -b / (2.0 * a)
            y0_val = f_polar(x0_val)
        # wir füllen per t weiter unten

        # pro Thermiktyp (t = 0..4)
        for t in range(5):
            # Recompute C_A_K und C_W_K analog MATLAB (x0 in km/h -> m/s conversion)
            if x0_val == -np.inf or np.isnan(x0_val):
                C_A_K = np.nan
                C_W_K = np.nan
            else:
                C_A_K = (Ga * g) / ((rho / 2.0) * (x0_val / 3.6)**2 * S)
                C_W_K = -C_A_K / ((x0_val / 3.6) / y0_val)

            # Thermik-Anteilfunktion (wA as function of r)
            def w_A_func(r_arr):
                return thermik["a"][t] + (r_arr**2) * thermik["b"][t]

            w_ST, w_A, w_SK, phi = berechne_w_ST(C_A_K, C_W_K, g, r, w_A_func, G, konst)

            # negative w_ST nicht zulassen (wie MATLAB)
            w_ST = np.maximum(w_ST, 0.0)

            
            # max und argmax
            idx_max = int(np.nanargmax(w_ST))
            max_w_ST[t, ui] = w_ST[idx_max]
            r_max[t, ui] = r[idx_max]
            phi_max[t, ui] = phi[idx_max]

            # M als 2/3 des Maximums (MATLAB)
            M = max_w_ST[t, ui] * 0.8

            # Diskriminanten-Funktion Delta(m) = (b - m)^2 - 4 a (c - M)
            # analytische Lösung: b - m = ± 2 sqrt(a*(c - M))
            # -> m = b ∓ 2 sqrt(a*(c - M))
            # Wir behandeln Fälle:
            C_val = (c - M)
            # numerisch robust:
            if a * C_val >= 0:
                sqrt_term = np.sqrt(a * C_val)
                # zwei Kandidaten
                m1 = b - 2.0 * sqrt_term
                m2 = b + 2.0 * sqrt_term
                # wähle denjenigen m, der "sinnvoll" ist:
                # In MATLAB wurde fzero(Delta, b) verwendet -> Startwert b.
                # Wir wählen die Lösung, die näher an b liegt.
                if abs(m1 - b) <= abs(m2 - b):
                    m_tan = m1
                else:
                    m_tan = m2
            else:
                # keine reelle Tangentenlösung -> Fallback: setze m_tan = b und warne
                warnings.warn(f"No real tangent root at G_a={Ga}, t={t}. Using fallback m=b.")
                m_tan = b

            # A,B,C definiert für Tangentenberechnung
            A = a
            B = b - m_tan
            Cc = c - M

            if A == 0:
                x_tan_val = np.nan
                y_tan_val = np.nan
            else:
                x_tan_val = abs(-B / (2.0 * A))
                y_tan_val = a * x_tan_val**2 + b * x_tan_val + c

            x_tan[t, ui] = x_tan_val
            y_tan[t, ui] = y_tan_val

            # GZ wie im MATLAB: x_tan / (-y_tan * 3.6)
            if y_tan_val == 0 or np.isnan(y_tan_val):
                GZ_val = np.nan
            else:
                GZ_val = (x_tan_val / (-y_tan_val * 3.6))
            GZ[t, ui] = GZ_val

            # Gleitgeschwindigkeit (konstant aus Skript)
            wA_Gleit = -0.8*ballern
            


            # Bereich x > 100
            mask = x > 100
            x_bereich = x[mask]
            y_bereich = y[mask]
            # finde x bei dem y ~= wA_Gleit
            idx_rel = np.argmin(np.abs(y_bereich - wA_Gleit))
            x_Gl[0, ui] = x_bereich[idx_rel]
            y_Gl[0, ui] = y_bereich[idx_rel]

            # Zeiten: t_st, t_v, t_X etc.
            # Achtung Einheiten: GZ in [km/(m/s?)] ... wir übernehmen Formel aus MATLAB
            # t_st in Minuten
            if not np.isnan(max_w_ST[t, ui]) and max_w_ST[t, ui] > 0 and GZ_val and max_w_ST[t, ui] != 0:
                t_st = (((thermik["frac"][t] * D) / GZ_val) / max_w_ST[t, ui]) / 60.0
            else:
                t_st = np.nan

            if x_tan_val and not np.isnan(x_tan_val) and x_tan_val != 0:
                t_v = ((thermik["frac"][t] * D) / (x_tan_val / 3.6)) / 60.0
            else:
                t_v = np.nan

            t_X[t, ui] = t_st + t_v

        # t_Gl (Gleitzonenanteil) (thermik.frac[4] ist GL)
        if x_Gl[0, ui] != 0:
            t_Gl = ((thermik["frac"][4] * D) / (x_Gl[0, ui] / 3.6)) / 60.0
        else:
            t_Gl = np.nan

        # Cross-Country Geschwindigkeit V_XC (mittlere)
        # siehe MATLAB: V_XC(u)=(D/(t_X(1,u)*60+t_X(2,u)*60+t_X(3,u)*60+t_X(4,u)*60+t_Gl(1,u)*60))*3.6;
        # t_X rows 0..3, plus t_Gl
        t_sum_seconds = 0.0
        for tt in range(4):
            if not np.isnan(t_X[tt, ui]):
                t_sum_seconds += t_X[tt, ui] * 60.0
        t_sum_seconds += (t_Gl * 60.0) if not np.isnan(t_Gl) else 0.0

        if t_sum_seconds > 0:
            V_XC[ui] = (D / t_sum_seconds) * 3.6
        else:
            V_XC[ui] = np.nan

        # thermikdurch (weighted average)
        denom = 1.0 - thermik["frac"][4]
        if denom != 0:
            thermikdurch[ui] = (max_w_ST[0, ui] * thermik["frac"][0] +
                                max_w_ST[1, ui] * thermik["frac"][1] +
                                max_w_ST[2, ui] * thermik["frac"][2] +
                                max_w_ST[3, ui] * thermik["frac"][3]) / denom
        else:
            thermikdurch[ui] = np.nan

    # Bestes Ergebnis finden
    idxV_XC = int(np.nanargmax(V_XC))
    maxV_XC = V_XC[idxV_XC]
    geschw_eigen = x0_val  # x0(indexV_XC) in MATLAB; einfache Übernahme



    if np.any(max_w_ST == 0):
        st.info("ℹ️ Hinweis: Thermik ist zu schwach")

    else:

        # Ausgabe-Tabelle (Thermikarten)
        thermikarten = thermik["type"]

        Steiggeschwindigkeit = [round(v, 2) for v in [max_w_ST[0, idxV_XC], max_w_ST[1, idxV_XC], max_w_ST[2, idxV_XC], max_w_ST[3, idxV_XC], wA_Gleit]]
        Kreisradius = [round(v, 1) for v in [r_max[0, idxV_XC], r_max[1, idxV_XC], r_max[2, idxV_XC], r_max[3, idxV_XC]]] + ["-"]
        Haengewinkel = [round(v, 0) for v in [phi_max[0, idxV_XC], phi_max[1, idxV_XC], phi_max[2, idxV_XC], phi_max[3, idxV_XC]]] + ["-"]
        Vorfluggeschwindigkeit = [round(v, 0) for v in [x_tan[0, idxV_XC], x_tan[1, idxV_XC], x_tan[2, idxV_XC], x_tan[3, idxV_XC], x_Gl[0, idxV_XC]]]
        Flugzeit = [round(v, 0) for v in [t_X[0, idxV_XC], t_X[1, idxV_XC], t_X[2, idxV_XC], t_X[3, idxV_XC], ((thermik["frac"][4] * D) / (x_Gl[0, idxV_XC] / 3.6)) / 60.0 if x_Gl[0, idxV_XC] != 0 else np.nan]]
        # Kurbelgeschwindigkeit-Berechnung: geschw_eigen * sqrt(1/cos(phi))
        def kurbel(geschw_eigen_local, phi_deg):
            try:
                return geschw_eigen_local * sqrt(1.0 / cos(radians(phi_deg)))
            except:
                return np.nan

        Kurbelgeschw = [
            round(kurbel(geschw_eigen, Haengewinkel[0]), 0),
            round(kurbel(geschw_eigen, Haengewinkel[1]), 0),
            round(kurbel(geschw_eigen, Haengewinkel[2]), 0),
            round(kurbel(geschw_eigen, Haengewinkel[3]), 0),
            "-"
        ]

        T = pd.DataFrame({
            "Steigen[m/s]": Steiggeschwindigkeit,
            "Radius[m]": Kreisradius,
            "Querlage[°]": Haengewinkel,
            "Kurbelgeschw.[km/h]": Kurbelgeschw,
            "Vorfluggeschw.[km/h]": Vorfluggeschwindigkeit,
            "Flugzeit[min]": Flugzeit
        }, index=thermikarten)

    

        total_time = 0.0
        for val in Flugzeit:
            if not np.isnan(val):
                total_time += val
        print(f"Gesamt Flugzeit (Summe der Blöcke) = {total_time:.2f} min (vereinfachte Summe)")
        if total_time > 0:
            schnitt = (D / (total_time / 60.0)) / 1000.0
            print(f"Schnitt = {schnitt:.2f} km/h (vereinfachte Rechnung)")
        else:
            print("Schnitt konnte nicht berechnet werden (Zeitdaten unvollständig).")

        total_time_h = total_time / 60.0

        # --- Ergebnis anzeigen ---
        st.subheader("Thermikvarianten")
        st.dataframe(T)  # Dein DataFrame T aus dem Originalcode
        st.markdown('<p style="font-size:12px;">Basierend auf der Index-Berechnung der Idaflieg</p>', unsafe_allow_html=True)



        st.subheader("Flugwerte")
        st.text(f"theoretisch erreichbare Überlandschnittgeschw. = {maxV_XC:.2f} km/h")
        st.text(f"Optimales Gesamtgewicht = {G_a_list[idxV_XC]} kg")        
        st.text(f"Thermikdurchschnitt = {thermikdurch[idxV_XC]:.2f} m/s")
        
        
        stunden = int(total_time_h)                     # ganze Stunden
        minuten = int((total_time_h - stunden) * 60)   # Rest in Minuten

        st.text(f"Gesamt Flugzeit = {stunden} h {minuten} min")
        st.text(f"Gesamt Flugzeit = {total_time:.1f} min")

    


