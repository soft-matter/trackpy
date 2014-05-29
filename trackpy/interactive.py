from __future__ import print_function

import warnings

# We will use IPython widgets and traitlets and some of the mpld3 internals.
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from IPython.html import widgets
from IPython.display import display, Javascript
from IPython.utils.traitlets import Unicode, Integer, Float

import mpld3
from mpld3._display import NumpyEncoder
from mpld3.urls import MPLD3_URL, D3_URL
from mpld3.utils import get_id
from mpld3.mplexporter import Exporter
from mpld3.mpld3renderer import MPLD3Renderer

import json
import random

from pandas import DataFrame


JAVASCRIPT = """
require(["widgets/js/widget"], function(WidgetManager){
    
    var fig;
    var FigureView = IPython.DOMWidgetView.extend({
        render: function(){
            this.$figure = $('<div />')
                .attr('id', this.model.get('figid'))
                .appendTo(this.$el);

            var that = this;
            
            // This must be called after the DOM is updated
            // to include the <div> for this view.
            var draw_plot =  function() {
                // Fill div with mpld3 figure.
                var figid = that.model.get('figid');
                var figure_json = JSON.parse(that.model.get('figure_json'));
                var extra_js = that.model.get('extra_js');
            
                if(typeof(window.mpld3) !== "undefined" && window.mpld3._mpld3IsLoaded){
                    !function (mpld3){
                            eval(extra_js);
                            that.fig = mpld3.draw_figure(figid, figure_json);
                    }(mpld3);
                } else {
                    var d3_url = that.model.get('d3_url');
                    var mpld3_url = that.model.get('mpld3_url');
                    require.config({paths: {d3: d3_url }});
                    require(["d3"], function(d3){
                        window.d3 = d3;
                        $.getScript(mpld3_url, function(){
                            eval(extra_js);
                            that.fig = mpld3.draw_figure(figid, figure_json);
                        });
                    });
                }
            };
            
            var handle_min_mass_change = function() {
                that.filter_points();
            };

            var handle_max_mass_change = function() {
                that.filter_points();
            };

            this.model.on("change:min_mass", handle_min_mass_change);
            this.model.on("change:max_mass", handle_max_mass_change);
            this.model.on("change:initialized", draw_plot);
            
        },
        
        filter_points: function() {
            var pts = mpld3.get_element(this.model.get('idpts'));
            var min_mass = this.model.get('min_mass');
            var max_mass = this.model.get('max_mass');
            var mass_data = JSON.parse(this.model.get('mass_data'));
            pts.elements().transition().style('opacity', function(d, i) { return (mass_data[i] > min_mass) & (mass_data[i] < max_mass) ? 1 : 0});
        },
        
        update: function() {
            
            return;
        
        },
        
    });
    WidgetManager.register_widget_view('FigureView', FigureView);
});
"""


def interactive_filter(f, image, imshow_kws={}, plot_kws={}):
    fig, ax = plt.subplots()

    plot_kws['ms'] = plot_kws.get('ms', 15)
    plot_kws['mew'] = plot_kws.get('mew', 2)
    plot_kws['mec'] = plot_kws.get('mec', 'r')
    plot_kws['mfc'] = plot_kws.get('mfc', 'none')
    plot_kws['marker'] = plot_kws.get('', 'o')
    plot_kws['ls'] = plot_kws.get('', 'none')

    imshow_kws['origin'] = imshow_kws.get('origin', 'lower')
    if imshow_kws['origin'] == 'upper':
        warnings.warn("Due to a bug in mpld3 v0.3, box zooming will not behave properly "
                      "origin='upper'.")
    
    points = ax.plot(f.x, f.y, **plot_kws)

    ax.imshow(image, **imshow_kws)

    renderer = MPLD3Renderer()
    Exporter(renderer, close_mpl=False).run(fig)
    fig, figure_json, extra_css, extra_js = renderer.finished_figures[0]

    my_widget = FigureWidget()
    my_widget.figure_json = json.dumps(figure_json, cls=NumpyEncoder)
    my_widget.extra_js = extra_js
    my_widget.extra_css = extra_css
    my_widget.figid = 'fig_' + get_id(fig) + str(int(random.random() * 1E10))
    my_widget.idpts = mpld3.utils.get_id(points[0], 'pts')
    my_widget.mass_data = json.dumps(f['mass'].tolist(), cls=NumpyEncoder)

    display(Javascript(JAVASCRIPT))
    fig.clf()

    from IPython.utils.traitlets import link
    for col in ['mass']:
        if np.issubdtype(f[col].dtype, np.float):
            slider_widget = widgets.IntSliderWidget
        elif np.issubdtype(f[col].dtype, np.integer):
            slider_widget = widgets.FloatSliderWidget
            step = 1
        else:
            continue
        if f[col].ptp() > 100:
            step = 1
        else:
            step = f[col].ptp()/100
        min_sw = slider_widget(min=f[col].min(), max=f[col].max(),
                               step=step,
                               description='minimum {0}'.format(col))
        max_sw = slider_widget(min=f[col].min(), max=f[col].max(),
                               step=step,
                               description='maximum {0}'.format(col))
        link((my_widget, 'min_{0}'.format(col)), (min_sw, 'value'))
        link((my_widget, 'max_{0}'.format(col)), (max_sw, 'value'))
        min_sw.value = f[col].min()
        max_sw.value = f[col].max()
        display(min_sw)
        display(max_sw)

    return my_widget


class FigureWidget(widgets.DOMWidget):
    _view_name = Unicode('FigureView', sync=True)

    mpld3_url = Unicode(MPLD3_URL, sync=True)  # TODO: Allow local mpld3 and d3.
    d3_url = Unicode(D3_URL[:-3], sync=True)

    figure_json = Unicode('', sync=True)  # to be filled in after instantiation
    extra_js = Unicode('', sync=True)  # for plugin support
    extra_css = Unicode('', sync=True)

    figid = Unicode('', sync=True)
    initialized = Unicode('', sync=True)  # used to trigger first drawing after DOM is updated

    mass_data = Unicode('', sync=True)
    min_mass = Float(0, sync=True)
    max_mass = Float(0, sync=True)
    idpts = Unicode('', sync=True)

    def display(self):
        display(self)
        self.initialized = ' '

    @property
    def conditional(self):
        "Return a string expressing conditionals reflected by the widget settings."
        c = "(mass > {0}) & (mass < {1})".format(self.min_mass, self.max_mass)
        return c
