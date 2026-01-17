package com.grocerybuddy.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.grocerybuddy.data.GroceryDatabase
import com.grocerybuddy.data.GroceryItem
import com.grocerybuddy.network.RobotWebSocket
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class RobotViewModel(application: Application) : AndroidViewModel(application) {

    private val database = GroceryDatabase.getDatabase(application)
    private val groceryDao = database.groceryDao()

    private val robotWebSocket = RobotWebSocket()

    val connectionState = robotWebSocket.connectionState.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5000),
        RobotWebSocket.ConnectionState.DISCONNECTED
    )

    val robotStatus = robotWebSocket.robotStatus.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5000),
        RobotWebSocket.RobotStatus()
    )

    val groceryList = groceryDao.getAllItems().stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5000),
        emptyList()
    )

    private val _showCalibrationSuccess = MutableStateFlow(false)
    val showCalibrationSuccess = _showCalibrationSuccess.asStateFlow()

    private val _showEmergencyStopAlert = MutableStateFlow(false)
    val showEmergencyStopAlert = _showEmergencyStopAlert.asStateFlow()

    init {
        connectToRobot()
    }

    // Robot Control Functions
    fun connectToRobot() {
        robotWebSocket.connect()
    }

    fun disconnectFromRobot() {
        robotWebSocket.disconnect()
    }

    fun reconnect() {
        robotWebSocket.reconnect()
    }

    fun calibrate() {
        robotWebSocket.calibrate()
        _showCalibrationSuccess.value = true
        viewModelScope.launch {
            kotlinx.coroutines.delay(2000)
            _showCalibrationSuccess.value = false
        }
    }

    fun startTracking() {
        robotWebSocket.startTracking()
    }

    fun stopTracking() {
        robotWebSocket.stopTracking()
    }

    fun emergencyStop() {
        robotWebSocket.emergencyStop()
        _showEmergencyStopAlert.value = true
        viewModelScope.launch {
            kotlinx.coroutines.delay(3000)
            _showEmergencyStopAlert.value = false
        }
    }

    fun setMode(mode: RobotWebSocket.CameraMode) {
        robotWebSocket.setMode(mode)
    }

    // Grocery List Functions
    fun addGroceryItem(name: String) {
        viewModelScope.launch {
            if (name.isNotBlank()) {
                groceryDao.insert(GroceryItem(name = name.trim()))
            }
        }
    }

    fun toggleItem(item: GroceryItem) {
        viewModelScope.launch {
            groceryDao.update(item.copy(isChecked = !item.isChecked))
        }
    }

    fun deleteItem(item: GroceryItem) {
        viewModelScope.launch {
            groceryDao.delete(item)
        }
    }

    fun deleteCheckedItems() {
        viewModelScope.launch {
            groceryList.value
                .filter { it.isChecked }
                .forEach { groceryDao.delete(it) }
        }
    }

    fun clearAllItems() {
        viewModelScope.launch {
            groceryDao.deleteAll()
        }
    }

    override fun onCleared() {
        super.onCleared()
        robotWebSocket.disconnect()
    }
}
