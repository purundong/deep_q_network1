#pragma once

#include <QtWidgets/QWidget>
#include "ui_deep_q_network.h"
#include <QMenu>
#include "state/environment.h"
#include <boost/thread.hpp>
#include <memory>
#include "qcustomplot.h"

class deep_q_network : public QWidget
{
	Q_OBJECT

public:
	deep_q_network(QWidget* parent = nullptr);
	~deep_q_network();
	void init_tableWidget();
	void show_environment();
signals:
	void sig_loss(double x, double y);
	void sig_init(double x_min, double x_max, double y_min, double y_max);
	void sig_msg_box(QString msg);
	void sig_show_environment();
public slots:
	void sot_loss(double x, double y);
	void sot_init(double x_min, double x_max, double y_min, double y_max);
	void sot_msg_box(QString msg);
	void sot_show_environment();
	void menu_set_rewards();
	void menu_set_transfers();
	void menu_state_info();
	void on_pushButton_make_state_clicked();
	void on_pushButton_dqn_clicked();
	void on_pushButton_qlearning_clicked();
	void on_horizontalSlider_valueChanged(int value);
	void on_tableWidget_customContextMenuRequested(QPoint point);
private:
	Ui::deep_q_networkClass ui;
	int _magni_fication = 10;
	QMenu* _menu{ nullptr };
private:
	environment_ptr _environment;
	std::unique_ptr<boost::thread> _thr;
	std::unique_ptr<QCustomPlot> _plot;
};
