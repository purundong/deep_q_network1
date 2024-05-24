#pragma once

#include <QtWidgets/QWidget>
#include "ui_deep_q_network.h"
#include <QMenu>
#include "state/environment.h"

class deep_q_network : public QWidget
{
	Q_OBJECT

public:
	deep_q_network(QWidget* parent = nullptr);
	~deep_q_network();
	void init_tableWidget();
public slots:
	void menu_set_rewards();
	void menu_set_transfers();
	void on_pushButton_make_state_clicked();
	void on_pushButton_solve_clicked();
private:
	Ui::deep_q_networkClass ui;
	int _magni_fication = 10;
	QMenu* _menu{ nullptr };
private:
	environment_ptr _environment;
};
