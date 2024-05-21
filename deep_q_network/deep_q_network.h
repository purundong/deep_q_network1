#pragma once

#include <QtWidgets/QWidget>
#include "ui_deep_q_network.h"

class deep_q_network : public QWidget
{
    Q_OBJECT

public:
    deep_q_network(QWidget *parent = nullptr);
    ~deep_q_network();

private:
    Ui::deep_q_networkClass ui;
};
