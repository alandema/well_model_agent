def FOWM(x, t, u, param):

    x1, x2, x3, x4, x5, x6 = x
    mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr = param
    u1, u2, u3, u4 = u

    Rol     = 900.0;
    R       = 8314.0;
    T       = 298.0;
    M       = 18.0;
    g       = 9.81;
    #teta    = tf.constant(m.pi)/4;
    Ps      = u3;
    Pr      = u4;
    ALFAgw  = 0.0188;
    Romres  = 891.9523;

    L       = 4497.0;
    Lt      = 1639.0;
    La      = 1118.0;
    D       = 0.152;
    Dt      = 0.150;
    Da      = 0.140;
    A       = 0.018145839167135;
    Vr      = 81.601838734604499;
    Vt      = 28.963520770689396;
    Va      = 17.210272874895608;
    Hvgl    = 916.0;
    Hpdg    = 1117.0;
    Ht      = 1279.0;

    Wgc    = u1*101325.0*M/(293.0*R)/3600.0/24.0;
    z     = u2/100.0;

    # RISER + PIPELINE
    Peb = x1*R*T/(M*Veb);
    Prt = x2*R*T/(M*(Vr-(x3+mlstill)/Rol));
    Prb = Prt + (x3 + mlstill)*g*0.7071/A;

    ALFAg = x2/(x2 + x3);
    ALFAl = 1.0 - ALFAg;

    Wout = Cout*z*np.sqrt(max(0,Rol*(Prt - Ps)))
    Wlout = ALFAl*Wout;
    Wgout = ALFAg*Wout;
    Wg = Cg*max(0,(Peb - Prb));

    # TUBING
    Vgt = Vt - x6/Rol;

    ROgt = x5/Vgt;
    ROmt = (x5 + x6)/Vt;

    Ptt  = ROgt*R*T/M;
    Ptb  = Ptt + ROmt*g*Hvgl;
    Ppdg = Ptb + Romres*g*(Hpdg-Hvgl);
    Pbh    = Ppdg + Romres*g*(Ht-Hpdg);

    ALFAgt = x5/(x6 + x5);

    Wwh = Kw*np.sqrt(ROmt*max(0,(Ptt - Prb)));
    Wwhg = Wwh*ALFAgt;
    Wwhl = Wwh*(1 - ALFAgt);
    Wr   = max(0, Kr*(1-0.2*Pbh/Pr-0.8*(Pbh/Pr)**2));

    # ANULAR

    Pai  = ((R*T/(Va*M)) + (g*La/Va))*x4;
    ROai = M*Pai/(R*T);
    Wiv  = Ka*np.sqrt(ROai*max(0,(Pai - Ptb)));

    # ODE
    dx1 = (1 - E)*(Wwhg) - Wg;                    # massa de gás na bolha
    dx2 = E*(Wwhg) + Wg - Wgout;                  # massa gás no riser
    dx3 = Wwhl - Wlout;                           # massa de líquido no riser
    dx4 = Wgc - Wiv;                              # massa de gás no anular
    dx5 = Wr * ALFAgw + Wiv - Wwhg;               # massa de gás no tubing
    dx6 = Wr * ( 1 - ALFAgw ) - Wwhl;

    dx = [dx1, dx2, dx3, dx4, dx5, dx6]

    return dx

def calcpres(x, u, param):
    x1, x2, x3, x4, x5, x6 = x
    mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr = param
    u1, u2, u3, u4 = u

    Rol     = 900.0;
    R       = 8314.0;
    T       = 298.0;
    M       = 18.0;
    g       = 9.81;
    #teta    = tf.constant(m.pi)/4;
    Ps      = u3;
    Pr      = u4;
    ALFAgw  = 0.0188;
    Romres  = 891.9523;

    L       = 4497.0;
    Lt      = 1639.0;
    La      = 1118.0;
    D       = 0.152;
    Dt      = 0.150;
    Da      = 0.140;
    A       = 0.018145839167135;
    Vr      = 81.601838734604499;
    Vt      = 28.963520770689396;
    Va      = 17.210272874895608;
    Hvgl    = 916.0;
    Hpdg    = 1117.0;
    Ht      = 1279.0;

    Wgc    = u1*101325.0*M/(293.0*R)/3600.0/24.0;
    z     = u2/100.0;

    # RISER + PIPELINE
    Peb = x1*R*T/(M*Veb);
    Prt = x2*R*T/(M*(Vr-(x3+mlstill)/Rol));
    Prb = Prt + (x3 + mlstill)*g*0.7071/A;

    ALFAg = x2/(x2 + x3);
    ALFAl = 1.0 - ALFAg;

    Wout = Cout*z*np.sqrt(max(0,Rol*(Prt - Ps)))
    Wlout = ALFAl*Wout;
    Wgout = ALFAg*Wout;
    Wg = Cg*max(0,(Peb - Prb));

    # TUBING
    Vgt = Vt - x6/Rol;

    ROgt = x5/Vgt;
    ROmt = (x5 + x6)/Vt;

    Ptt  = ROgt*R*T/M;
    Ptb  = Ptt + ROmt*g*Hvgl;
    Ppdg = Ptb + Romres*g*(Hpdg-Hvgl);

    return Ppdg, Ptt, Prt, Prb