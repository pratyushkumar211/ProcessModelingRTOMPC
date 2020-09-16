#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [depends] hvacnntest.mat
# [makes] figure
"""
Created on Wed Oct 10 18:21:30 2018
@author: pkumar
"""
import custompath
custompath.add()

# Importing the required packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plottools.matio import loadmat
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import plottools

########### Setting for plots ############
ms=0.9
lw=0.5
yax=-0.25


# Figure Size
figx=10
figy=8

#White Spacing
ws=1.2
hs=0.8

# Grid Spacing
gsy=4
gsx=4

# fonts of plots
plt.rcParams.update({'font.size': 16})

# Labels for X and Y axis
ylabel=['$T_{A1}$ (K)','$T_{C1}$ (K)','$T_{A2}$ (K)','$T_{C2}$ (K)']
ulabel=['$v_{1}$','$v_{2}$']
xlabel = 'Time (hr)'

# Do not rotate ylabel unless required
ylabelRotation = False

# Color of plots
ucolor = 'g'
ycolor = 'b-'
xspcolor = 'r'

# Colors in the ID data color
plantcolor = 'r'
lcolor = 'k'
nncolor = 'b'

# Colors in the gain response plots
nonlincolor = 'r'
lincolor = 'b'
sscolor = 'k'
sslstyle = '--'

# legends for the gain response plots
linlabel = 'Linear model'
plantlabel = 'Plant'


# Limits on plots which use valve positions
vlim = [0, 0.7]
###########################################

def iddata(x, xlin, xnn, u, time):
    

    gs=gridspec.GridSpec(gsy,gsx)
    gs.update(wspace=ws, hspace=hs)
    fig1=plt.figure(figsize=(figx,figy))
        
    # Plot the measurements
    Nx = x.shape[0]
    for i in range(0, Nx):
        
        ax=fig1.add_subplot(gs[i,gsx-2:gsx]) 
        ax.plot(time[0:tfrac], x[i,0:tfrac], plantcolor, linewidth = lw)
        ax.plot(time[0:tfrac], xlin[i,0:tfrac], lcolor, linewidth = lw)
        ax.plot(time[0:tfrac], xnn[i,0:tfrac], nncolor, linewidth = lw)
        ax.set_ylabel(ylabel[i])
        ax.set_xlim(min(time[0:tfrac]),max(time[0:tfrac]))
        ax.get_yaxis().set_label_coords(yax,0.5)
    
    ax.set_xlabel(xlabel)

    # Plot the Actions
    Nu = u.shape[0]
    for i in range(0, Nu):
        
        ax=fig1.add_subplot(gs[2*i:2*i+2,0:gsx-2]) 
        ax.step(time[0:tfrac], u[i,0:tfrac], ucolor, linewidth=lw)
        ax.set_ylabel(ulabel[i], rotation = ylabelRotation)
        ax.set_xlim(min(time[0:tfrac]),max(time[0:tfrac]))
        ax.get_yaxis().set_label_coords(yax,0.5)
        ax.set_ylim([0, 1])
    ax.set_xlabel(xlabel)

        
    return fig1

def closedLoop(y, u, ysp, time, controller):
    

    gs=gridspec.GridSpec(gsy,gsx)
    gs.update(wspace=ws, hspace=hs)
    fig1=plt.figure(figsize=(figx,figy))
        
    # Plot the measurements
    Nx = y.shape[0]
    for i in range(0, Nx):
        
        ax=fig1.add_subplot(gs[i,gsx-2:gsx]) 
        ax.plot(time[0:tfrac], y[i,0:tfrac], ycolor, markersize=ms)
        ax.plot(time[0:tfrac], ysp[i,0:tfrac], xspcolor, linewidth=lw)
        ax.set_ylabel(ylabel[i])
        ax.set_xlim(min(time[0:tfrac]),max(time[0:tfrac]))
        ax.get_yaxis().set_label_coords(yax,0.5)
    
    ax.set_xlabel(xlabel)

    # Plot the Actions
    Nu = u.shape[0]
    for i in range(0, Nu):
        
        ax=fig1.add_subplot(gs[2*i:2*i+2,0:gsx-2]) 
        ax.step(time[0:tfrac], u[i,0:tfrac], ucolor, linewidth=lw)
        ax.set_ylabel(ulabel[i], rotation = ylabelRotation)
        ax.set_xlim(min(time[0:tfrac]),max(time[0:tfrac]))
        ax.get_yaxis().set_label_coords(yax,0.5)        
        ax.set_ylim(vlim)

    ax.set_xlabel(xlabel)        
    if plottools.isPresentation():
        fig1.suptitle(controller)
    
    return fig1


# Creating a pdf 
pdf_pages = PdfPages('plotshvacnn.pdf')

### Figure 1: Model Fit #####
data = loadmat("hvacnntest.mat",asrecarray=True,squeeze=True)
x = data['x']
xlin = data['xlin']
xnnp = data['xnnp']
u = data['u']
time = data['tpred']
tfrac = 60
fig = iddata(x, xlin, xnnp, u, time)
pdf_pages.savefig(fig, bbox_inches="tight")

### Figure 2: Closed loop linear ID + integrating dist ####
ycl = data['ycl']
ucl = data['ucl']
ysp = data['ysp']
ti = data['ti']
Ntr = int(data['Ntr'])
del data
tfrac = Ntr
fig = closedLoop(ycl['lin'], ucl['lin'], ysp, ti, 'Linear MPC')
pdf_pages.savefig(fig, bbox_inches="tight")

### Figure 3: Closed loop Nonlinear NN + integrating dist ####
fig = closedLoop(ycl['nn'], ucl['nn'], ysp, ti, 'NN-based MPC')
pdf_pages.savefig(fig, bbox_inches="tight")

### Figure 4: Nominal Controller ####
fig = closedLoop(ycl['nom'], ucl['nom'], ysp, ti, 'Nominal MPC')
pdf_pages.savefig(fig, bbox_inches="tight")

### Figure 5: 4 by 4 plot of xs vs us response ##########
data=loadmat("hvacnndata.mat",asrecarray=True,squeeze=True)
uvar = data['uvar']
uxlin = data['xslin']
uxnonlin = data['xsnonlin']
xs = data['xs']
us = data['us']
ysp1 = data['ysp1']
ysp2 = data['ysp2']
del data

gs = gridspec.GridSpec(gsy, gsx)
gs.update(wspace = ws, hspace = hs)
fig = plt.figure(figsize = (figx, figy))
xlim = [min(uvar),max(uvar)]
    
# Plot the measurements
for i in range(0, 2):
    
    for j in range(0,2):
        
        # Create axis and font settings
        ax=fig.add_subplot(gs[2*i:2*i+2, 2*j:2*j+2])        
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Add plots
        ax.plot(uvar, uxlin[2*i,:,j], lincolor, linewidth=lw, label = linlabel)
        ax.plot(uvar, uxnonlin[2*i,:,j], nonlincolor, linewidth=lw, label = plantlabel)
        ax.legend()
        
        # Mark the steady states
        ylim = ax.get_ylim()
        yfrac = (xs[2*j]-ylim[0])/(ylim[1]-ylim[0])
        ax.axvline(us[j], ymin = 0, ymax = yfrac, color = sscolor, linestyle = sslstyle)
        ax.axhline(xs[2*i], xmin = 0, xmax = us[j], color = sscolor, linestyle = sslstyle)
        ax.set_xlim(xlim)

        # Labels
        ax.set_ylabel(ylabel[2*i], rotation = ylabelRotation)
        ax.set_xlabel(ulabel[j])
        
        # Extra point on x and y axis
        xticks = [xlim[0] , us[j], xlim[1]]
        ax.set_xticks(xticks)
        yticks = [ylim[0], xs[2*j], ylim[1]]
        ax.set_yticks(yticks)
    
pdf_pages.savefig(fig, bbox_inches="tight")


# Close PDF
pdf_pages.close()