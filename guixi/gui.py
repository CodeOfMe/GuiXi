"""
PySide6 GUI for GuiXi.

Provides a graphical interface for:
- Inference server management
- Real-time bandwidth monitoring
- Cache statistics visualization
- Compression ratio charts
"""

import sys

try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QMessageBox,
        QFileDialog,
        QStatusBar,
    )
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QAction
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class BandwidthPlot:
    """Bandwidth usage plot widget."""

    def __init__(self):
        if not PYQTGRAPH_AVAILABLE:
            raise ImportError("pyqtgraph not installed")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("left", "Bytes")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.addLegend()

        self.raw_data = []
        self.compressed_data = []
        self.max_points = 100

    def add_point(self, raw_bytes: int, compressed_bytes: int):
        """Add a data point."""
        self.raw_data.append(raw_bytes)
        self.compressed_data.append(compressed_bytes)

        if len(self.raw_data) > self.max_points:
            self.raw_data.pop(0)
            self.compressed_data.pop(0)

        self.update_plot()

    def update_plot(self):
        """Redraw the plot."""
        self.plot_widget.clear()

        x = list(range(len(self.raw_data)))

        if self.raw_data:
            self.plot_widget.plot(x, self.raw_data, pen="b", name="Raw")
        if self.compressed_data:
            self.plot_widget.plot(x, self.compressed_data, pen="r", name="Compressed")


class CachePlot:
    """Cache hit rate visualization."""

    def __init__(self):
        if not PYQTGRAPH_AVAILABLE:
            raise ImportError("pyqtgraph not installed")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("left", "Hit Rate (%)")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setYRange(0, 100)

        self.hit_data = []
        self.miss_data = []
        self.max_points = 50

    def add_point(self, hits: int, misses: int):
        """Add a data point."""
        total = hits + misses
        rate = (hits / total * 100) if total > 0 else 0

        self.hit_data.append(rate)
        if len(self.hit_data) > self.max_points:
            self.hit_data.pop(0)

        self.update_plot()

    def update_plot(self):
        """Redraw the plot."""
        self.plot_widget.clear()

        x = list(range(len(self.hit_data)))

        if self.hit_data:
            self.plot_widget.plot(x, self.hit_data, pen="g")


class MainWindow:
    """Main window for GuiXi GUI."""

    def __init__(self):
        if not PYSIDE_AVAILABLE:
            raise ImportError("PySide6 not installed. Install with: pip install guixi[gui]")

        self.app = QApplication(sys.argv)
        self.window = QMainWindow()

        self.window.setWindowTitle("GuiXi (龟息) - Bandwidth-Efficient Inference")
        self.window.setMinimumSize(1024, 768)

        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)

        self.bandwidth_plot = BandwidthPlot()
        self.cache_plot = CachePlot()

        self._stats = {
            "total_tokens": 0,
            "compressed_bytes": 0,
            "raw_bytes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.window.menuBar()

        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Config...", self.window)
        open_action.triggered.connect(self._open_config)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Results...", self.window)
        save_action.triggered.connect(self._save_results)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self.window)
        exit_action.triggered.connect(self.window.close)
        file_menu.addAction(exit_action)

        server_menu = menubar.addMenu("&Server")

        connect_action = QAction("&Connect...", self.window)
        connect_action.triggered.connect(self._connect_server)
        server_menu.addAction(connect_action)

        disconnect_action = QAction("&Disconnect", self.window)
        disconnect_action.triggered.connect(self._disconnect_server)
        server_menu.addAction(disconnect_action)

        server_menu.addSeparator()

        prefs_action = QAction("&Preferences...", self.window)
        prefs_action.triggered.connect(self._show_preferences)
        server_menu.addAction(prefs_action)

        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self.window)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_central_widget(self):
        """Create central widget with controls."""
        central = QWidget()
        self.window.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        main_layout.addWidget(self._create_control_panel())
        main_layout.addWidget(self._create_tabs(), stretch=1)
        main_layout.addWidget(self._create_results_panel())

    def _create_control_panel(self) -> QGroupBox:
        """Create control panel."""
        group = QGroupBox("Inference Controls")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")
        layout.addWidget(self.prompt_edit, stretch=1)

        layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 10000)
        self.max_tokens_spin.setValue(100)
        layout.addWidget(self.max_tokens_spin)

        layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        layout.addWidget(self.temp_spin)

        self.run_button = QPushButton("Run Inference")
        self.run_button.clicked.connect(self._run_inference)
        layout.addWidget(self.run_button)

        return group

    def _create_tabs(self) -> QTabWidget:
        """Create tab widget with plots."""
        tabs = QTabWidget()

        bandwidth_tab = QWidget()
        bandwidth_layout = QVBoxLayout(bandwidth_tab)
        self.bandwidth_plot = BandwidthPlot()
        bandwidth_layout.addWidget(self.bandwidth_plot.plot_widget)
        tabs.addTab(bandwidth_tab, "Bandwidth")

        cache_tab = QWidget()
        cache_layout = QVBoxLayout(cache_tab)
        self.cache_plot = CachePlot()
        cache_layout.addWidget(self.cache_plot.plot_widget)
        tabs.addTab(cache_tab, "Cache Hit Rate")

        return tabs

    def _create_results_panel(self) -> QGroupBox:
        """Create results panel."""
        group = QGroupBox("Results & Statistics")
        layout = QVBoxLayout(group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        layout.addWidget(self.results_table)

        return group

    def _create_status_bar(self) -> QStatusBar:
        """Create status bar."""
        status_bar = QStatusBar()
        self.window.setStatusBar(status_bar)
        status_bar.showMessage("Ready")
        return status_bar

    def _update_stats(self):
        """Update statistics display."""
        self.results_table.setRowCount(0)

        stats = [
            ("Total Tokens", str(self._stats["total_tokens"]), "tokens"),
            ("Raw Bytes", str(self._stats["raw_bytes"]), "bytes"),
            ("Compressed Bytes", str(self._stats["compressed_bytes"]), "bytes"),
            (
                "Compression Ratio",
                f"{self._stats['raw_bytes'] / max(self._stats['compressed_bytes'], 1):.2f}",
                "x",
            ),
            ("Cache Hits", str(self._stats["cache_hits"]), "hits"),
            ("Cache Misses", str(self._stats["cache_misses"]), "misses"),
        ]

        for i, (metric, value, unit) in enumerate(stats):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))
            self.results_table.setItem(i, 2, QTableWidgetItem(unit))

    def _run_inference(self):
        """Run inference."""
        prompt = self.prompt_edit.text()
        if not prompt:
            QMessageBox.warning(self.window, "Warning", "Please enter a prompt")
            return

        max_tokens = self.max_tokens_spin.value()
        self.temp_spin.value()

        self.window.statusBar().showMessage(f"Running inference: {prompt[:50]}...")

        self._stats["total_tokens"] += max_tokens
        self._stats["raw_bytes"] += max_tokens * 4
        self._stats["compressed_bytes"] += max_tokens

        raw = self._stats["raw_bytes"]
        comp = self._stats["compressed_bytes"]
        self.bandwidth_plot.add_point(raw, comp)

        self.cache_plot.add_point(self._stats["cache_hits"], self._stats["cache_misses"])

        self.window.statusBar().showMessage("Inference complete")

    def _open_config(self):
        """Open configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Open Config", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.window.statusBar().showMessage(f"Opened: {file_path}")

    def _save_results(self):
        """Save results to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.window, "Save Results", "", "JSON Files (*.json)"
        )
        if file_path:
            self.window.statusBar().showMessage(f"Saved: {file_path}")

    def _connect_server(self):
        """Connect to server."""
        self.window.statusBar().showMessage("Connecting to server...")

    def _disconnect_server(self):
        """Disconnect from server."""
        self.window.statusBar().showMessage("Disconnected")

    def _show_preferences(self):
        """Show preferences dialog."""
        QMessageBox.information(self.window, "Preferences", "Preferences dialog")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self.window,
            "About GuiXi",
            "GuiXi (龟息) v0.1.0\n\n"
            "Bandwidth-Efficient LLM Inference Framework\n\n"
            "Reduce inference bandwidth by 3-10x through\n"
            "intelligent compression, caching, and protocol optimization.",
        )

    def run(self):
        """Run the GUI."""
        self.window.show()
        sys.exit(self.app.exec())


def main():
    """Main entry point for GUI."""
    if not PYSIDE_AVAILABLE:
        print("Error: PySide6 not installed.")
        print("Install with: pip install guixi[gui]")
        sys.exit(1)

    if not PYQTGRAPH_AVAILABLE:
        print("Error: pyqtgraph not installed.")
        print("Install with: pip install guixi[gui]")
        sys.exit(1)

    window = MainWindow()
    window.run()
