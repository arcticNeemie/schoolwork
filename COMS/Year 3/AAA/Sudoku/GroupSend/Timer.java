public class Timer {
	long startTime;
	long endTime;

	public void start() {
		startTime = System.nanoTime();
	}
	public void stop() {
		endTime = System.nanoTime();
	}
	public long getTime() {
		return (endTime - startTime);
	}
}
